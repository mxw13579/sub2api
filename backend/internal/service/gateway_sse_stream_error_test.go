package service

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// 本文件覆盖两个限流问题：
//   - 上游 HTTP 200 + SSE event:error 中的 rate_limit_error 被当成 403 处理、账号不冷却；
//   - Anthropic 429 不读 Retry-After，一律走兜底冷却。

// sseRateLimitRepoStub 记录限流 / 过载 / 临时不可调度写入。
type sseRateLimitRepoStub struct {
	AccountRepository
	rateLimitCalls     int
	lastRateLimitID    int64
	lastRateLimitReset time.Time
	overloadCalls      int
	tempCalls          int
	sessionWindowCalls int
}

func (r *sseRateLimitRepoStub) SetRateLimited(_ context.Context, id int64, resetAt time.Time) error {
	r.rateLimitCalls++
	r.lastRateLimitID = id
	r.lastRateLimitReset = resetAt
	return nil
}

func (r *sseRateLimitRepoStub) SetOverloaded(context.Context, int64, time.Time) error {
	r.overloadCalls++
	return nil
}

func (r *sseRateLimitRepoStub) SetTempUnschedulable(context.Context, int64, time.Time, string) error {
	r.tempCalls++
	return nil
}

func (r *sseRateLimitRepoStub) SetModelRateLimit(context.Context, int64, string, time.Time, ...string) error {
	return nil
}

func (r *sseRateLimitRepoStub) UpdateSessionWindow(context.Context, int64, *time.Time, *time.Time, string) error {
	r.sessionWindowCalls++
	return nil
}

// newMeridianAccountForSSETest 模拟线上的 Meridian 上游：Anthropic 平台 API Key 账号，未开启透传。
func newMeridianAccountForSSETest() *Account {
	return &Account{
		ID:          601,
		Name:        "meridian-account-01",
		Platform:    PlatformAnthropic,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key":  "meridian-key",
			"base_url": "https://meridian.example.com",
		},
		Status:      StatusActive,
		Schedulable: true,
	}
}

func newSSEStreamErrorTestService(repo AccountRepository, upstream HTTPUpstream) *GatewayService {
	cfg := &config.Config{Gateway: config.GatewayConfig{MaxLineSize: defaultMaxLineSize}}
	return &GatewayService{
		cfg:                  cfg,
		responseHeaderFilter: compileResponseHeaderFilter(cfg),
		httpUpstream:         upstream,
		rateLimitService:     NewRateLimitService(repo, nil, cfg, nil, nil),
		deferredService:      &DeferredService{},
	}
}

func newSSEStreamErrorTestContext() (*gin.Context, *httptest.ResponseRecorder) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	return c, rec
}

func sseStreamErrorOpsEvents(t *testing.T, c *gin.Context) []*OpsUpstreamErrorEvent {
	t.Helper()
	raw, ok := c.Get(OpsUpstreamErrorsKey)
	require.True(t, ok, "stream_error ops event must be recorded")
	events, ok := raw.([]*OpsUpstreamErrorEvent)
	require.True(t, ok)
	return events
}

func requireResetAround(t *testing.T, got time.Time, before, after time.Time, d time.Duration) {
	t.Helper()
	const tolerance = 2 * time.Second
	require.False(t, got.Before(before.Add(d-tolerance)), "reset %v earlier than expected %v", got, before.Add(d))
	require.False(t, got.After(after.Add(d+tolerance)), "reset %v later than expected %v", got, after.Add(d))
}

const meridianRateLimitErrorJSON = `{"type":"error","error":{"type":"rate_limit_error","message":"Claude Max rate limit reached. Wait a moment and try again.","retry_after":15742}}`

// 场景 1：rate_limit_error + retry_after，未输出 → 429、运维记录 429、按 retry_after 冷却。
func TestSSEStreamError_RateLimitBeforeOutput_Returns429AndCoolsDownByRetryAfter(t *testing.T) {
	c, _ := newSSEStreamErrorTestContext()
	repo := &sseRateLimitRepoStub{}
	svc := newSSEStreamErrorTestService(repo, nil)
	account := newMeridianAccountForSSETest()
	resp := &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": []string{"text/event-stream"}}}

	before := time.Now()
	failoverErr := svc.sseStreamErrorEventFailover(context.Background(), c, account, resp, meridianRateLimitErrorJSON, "claude-sonnet-4-5", false)
	after := time.Now()

	require.NotNil(t, failoverErr)
	require.Equal(t, http.StatusTooManyRequests, failoverErr.StatusCode)
	require.JSONEq(t, meridianRateLimitErrorJSON, string(failoverErr.ResponseBody))

	events := sseStreamErrorOpsEvents(t, c)
	require.Len(t, events, 1)
	require.Equal(t, "stream_error", events[0].Kind)
	require.Equal(t, http.StatusTooManyRequests, events[0].UpstreamStatusCode)

	require.Equal(t, 1, repo.rateLimitCalls)
	require.Equal(t, account.ID, repo.lastRateLimitID)
	requireResetAround(t, repo.lastRateLimitReset, before, after, 15742*time.Second)
	require.Zero(t, repo.overloadCalls)
	require.Empty(t, resp.Header.Get("Retry-After"), "synthetic Retry-After must not leak into the upstream response header")
}

// 场景 1（端到端）：经过 Forward 完整流式路径，Meridian 返回 200 + SSE rate_limit_error。
func TestSSEStreamError_ForwardRateLimitBeforeOutput_FailoverWith429(t *testing.T) {
	c, rec := newSSEStreamErrorTestContext()
	body := []byte(`{"model":"claude-sonnet-4-5","stream":true,"messages":[{"role":"user","content":"hello"}]}`)
	parsed, err := ParseGatewayRequest(NewRequestBodyRef(body), PlatformAnthropic)
	require.NoError(t, err)

	upstream := &anthropicHTTPUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader("event: error\ndata: " + meridianRateLimitErrorJSON + "\n\n")),
	}}
	repo := &sseRateLimitRepoStub{}
	svc := newSSEStreamErrorTestService(repo, upstream)

	before := time.Now()
	result, err := svc.Forward(context.Background(), c, newMeridianAccountForSSETest(), parsed)
	after := time.Now()
	require.Nil(t, result)

	var failoverErr *UpstreamFailoverError
	require.ErrorAs(t, err, &failoverErr)
	require.Equal(t, http.StatusTooManyRequests, failoverErr.StatusCode)
	require.Empty(t, rec.Body.String(), "pre-output rate limit must remain eligible for account failover")
	require.Equal(t, 1, repo.rateLimitCalls)
	requireResetAround(t, repo.lastRateLimitReset, before, after, 15742*time.Second)
}

// 场景 2：同上但已输出 → 冷却照常执行；返回值保持原 403，
// 调用方（三个 handler）在已输出时直接走 *FailoverExhausted、不会重放到其他账号。
func TestSSEStreamError_RateLimitAfterOutput_CoolsDownWithoutChangingReturnStatus(t *testing.T) {
	c, rec := newSSEStreamErrorTestContext()
	body := []byte(`{"model":"claude-sonnet-4-5","stream":true,"messages":[{"role":"user","content":"hello"}]}`)
	parsed, err := ParseGatewayRequest(NewRequestBodyRef(body), PlatformAnthropic)
	require.NoError(t, err)

	fixture := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n" +
		"event: error\ndata: " + meridianRateLimitErrorJSON + "\n\n"
	upstream := &anthropicHTTPUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(fixture)),
	}}
	repo := &sseRateLimitRepoStub{}
	svc := newSSEStreamErrorTestService(repo, upstream)

	before := time.Now()
	result, err := svc.Forward(context.Background(), c, newMeridianAccountForSSETest(), parsed)
	after := time.Now()
	require.Nil(t, result)

	var failoverErr *UpstreamFailoverError
	require.ErrorAs(t, err, &failoverErr)
	require.Equal(t, http.StatusForbidden, failoverErr.StatusCode, "post-output return status must stay unchanged")
	require.Contains(t, rec.Body.String(), "message_start", "output must have started")

	require.Equal(t, 1, repo.rateLimitCalls, "rate limit cooldown must apply even after output started")
	requireResetAround(t, repo.lastRateLimitReset, before, after, 15742*time.Second)

	events := sseStreamErrorOpsEvents(t, c)
	require.Equal(t, http.StatusTooManyRequests, events[len(events)-1].UpstreamStatusCode)
}

// 场景 3：rate_limit_error 无 retry_after、无头 → 429，走兜底冷却。
func TestSSEStreamError_RateLimitWithoutRetryAfter_UsesFallback(t *testing.T) {
	c, _ := newSSEStreamErrorTestContext()
	repo := &sseRateLimitRepoStub{}
	svc := newSSEStreamErrorTestService(repo, nil)
	resp := &http.Response{StatusCode: http.StatusOK, Header: http.Header{}}

	const errorJSON = `{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}`
	before := time.Now()
	failoverErr := svc.sseStreamErrorEventFailover(context.Background(), c, newMeridianAccountForSSETest(), resp, errorJSON, "claude-sonnet-4-5", false)
	after := time.Now()

	require.Equal(t, http.StatusTooManyRequests, failoverErr.StatusCode)
	require.Equal(t, 1, repo.rateLimitCalls)
	// 未注入 settingService 时兜底为 defaultRateLimit429CooldownSeconds。
	requireResetAround(t, repo.lastRateLimitReset, before, after, time.Duration(defaultRateLimit429CooldownSeconds)*time.Second)
}

// 场景 4：响应头 Retry-After 与正文 retry_after 同时存在 → 以响应头为准。
func TestSSEStreamError_RateLimitHeaderRetryAfterWinsOverBody(t *testing.T) {
	c, _ := newSSEStreamErrorTestContext()
	repo := &sseRateLimitRepoStub{}
	svc := newSSEStreamErrorTestService(repo, nil)
	resp := &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Retry-After": []string{"60"}}}

	const errorJSON = `{"type":"error","error":{"type":"rate_limit_error","message":"rate limited","retry_after":999}}`
	before := time.Now()
	failoverErr := svc.sseStreamErrorEventFailover(context.Background(), c, newMeridianAccountForSSETest(), resp, errorJSON, "claude-sonnet-4-5", false)
	after := time.Now()

	require.Equal(t, http.StatusTooManyRequests, failoverErr.StatusCode)
	require.Equal(t, 1, repo.rateLimitCalls)
	requireResetAround(t, repo.lastRateLimitReset, before, after, 60*time.Second)
}

// 场景 5：overloaded_error 未输出 / 已输出 → 与修改前一致（529 + 冷却 / 403 + 不处理）。
func TestSSEStreamError_OverloadedKeepsExistingBehavior(t *testing.T) {
	const errorJSON = `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded","retry_after":30}}`

	t.Run("before output", func(t *testing.T) {
		c, _ := newSSEStreamErrorTestContext()
		repo := &sseRateLimitRepoStub{}
		svc := newSSEStreamErrorTestService(repo, nil)
		resp := &http.Response{StatusCode: http.StatusOK, Header: http.Header{}}

		failoverErr := svc.sseStreamErrorEventFailover(context.Background(), c, newMeridianAccountForSSETest(), resp, errorJSON, "claude-sonnet-4-5", false)
		require.Equal(t, 529, failoverErr.StatusCode)
		require.Equal(t, 1, repo.overloadCalls)
		require.Zero(t, repo.rateLimitCalls)
		require.Equal(t, 529, sseStreamErrorOpsEvents(t, c)[0].UpstreamStatusCode)
	})

	t.Run("after output", func(t *testing.T) {
		c, _ := newSSEStreamErrorTestContext()
		repo := &sseRateLimitRepoStub{}
		svc := newSSEStreamErrorTestService(repo, nil)
		resp := &http.Response{StatusCode: http.StatusOK, Header: http.Header{}}

		failoverErr := svc.sseStreamErrorEventFailover(context.Background(), c, newMeridianAccountForSSETest(), resp, errorJSON, "claude-sonnet-4-5", true)
		require.Equal(t, http.StatusForbidden, failoverErr.StatusCode)
		require.Zero(t, repo.overloadCalls)
		require.Zero(t, repo.rateLimitCalls)
		require.Zero(t, repo.tempCalls)
		require.Equal(t, http.StatusForbidden, sseStreamErrorOpsEvents(t, c)[0].UpstreamStatusCode)
	})
}

// 场景 6：其他 error.type（如 api_error）→ 仍是 403，不冷却。
func TestSSEStreamError_OtherErrorTypesStay403WithoutCooldown(t *testing.T) {
	for _, errType := range []string{"api_error", "permission_error", ""} {
		for _, outputStarted := range []bool{false, true} {
			t.Run(errType+"/outputStarted="+strconv.FormatBool(outputStarted), func(t *testing.T) {
				c, _ := newSSEStreamErrorTestContext()
				repo := &sseRateLimitRepoStub{}
				svc := newSSEStreamErrorTestService(repo, nil)
				resp := &http.Response{StatusCode: http.StatusOK, Header: http.Header{}}

				errorJSON := `{"type":"error","error":{"type":` + strconv.Quote(errType) + `,"message":"boom","retry_after":60}}`
				failoverErr := svc.sseStreamErrorEventFailover(context.Background(), c, newMeridianAccountForSSETest(), resp, errorJSON, "claude-sonnet-4-5", outputStarted)
				require.Equal(t, http.StatusForbidden, failoverErr.StatusCode)
				require.Zero(t, repo.rateLimitCalls)
				require.Zero(t, repo.overloadCalls)
				require.Zero(t, repo.tempCalls)
				require.Equal(t, http.StatusForbidden, sseStreamErrorOpsEvents(t, c)[0].UpstreamStatusCode)
			})
		}
	}
}

func TestSetRetryAfterFromSSEErrorBody(t *testing.T) {
	cases := []struct {
		name     string
		existing string
		body     string
		want     string
	}{
		{name: "integer seconds", body: `{"error":{"retry_after":15742}}`, want: "15742"},
		{name: "fractional rounds up", body: `{"error":{"retry_after":1.2}}`, want: "2"},
		{name: "numeric string", body: `{"error":{"retry_after":"30"}}`, want: "30"},
		{name: "existing header kept", existing: "60", body: `{"error":{"retry_after":999}}`, want: "60"},
		{name: "missing", body: `{"error":{}}`, want: ""},
		{name: "zero", body: `{"error":{"retry_after":0}}`, want: ""},
		{name: "negative", body: `{"error":{"retry_after":-5}}`, want: ""},
		{name: "non numeric", body: `{"error":{"retry_after":"soon"}}`, want: ""},
		{name: "object", body: `{"error":{"retry_after":{"s":1}}}`, want: ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			hdr := http.Header{}
			if tc.existing != "" {
				hdr.Set("Retry-After", tc.existing)
			}
			setRetryAfterFromSSEErrorBody(hdr, []byte(tc.body))
			require.Equal(t, tc.want, hdr.Get("Retry-After"))
		})
	}
}

// 场景 7：handle429，Anthropic 账号只带 Retry-After: 120 → SetRateLimited(now+120s)，不走兜底。
func TestHandle429_AnthropicRetryAfterOnly(t *testing.T) {
	repo := &sseRateLimitRepoStub{}
	svc := NewRateLimitService(repo, nil, &config.Config{}, nil, nil)
	account := newMeridianAccountForSSETest()

	before := time.Now()
	svc.handle429(context.Background(), account, http.Header{"Retry-After": []string{"120"}}, []byte(`{"error":{"type":"rate_limit_error"}}`))
	after := time.Now()

	require.Equal(t, 1, repo.rateLimitCalls)
	require.Equal(t, account.ID, repo.lastRateLimitID)
	requireResetAround(t, repo.lastRateLimitReset, before, after, 120*time.Second)
}

func TestHandle429_AnthropicRetryAfterHTTPDate(t *testing.T) {
	repo := &sseRateLimitRepoStub{}
	svc := NewRateLimitService(repo, nil, &config.Config{}, nil, nil)

	target := time.Now().Add(10 * time.Minute).UTC().Truncate(time.Second)
	svc.handle429(context.Background(), newMeridianAccountForSSETest(),
		http.Header{"Retry-After": []string{target.Format(http.TimeFormat)}}, nil)

	require.Equal(t, 1, repo.rateLimitCalls)
	require.True(t, repo.lastRateLimitReset.Equal(target), "got %v want %v", repo.lastRateLimitReset, target)
}

// 场景 8：同时带 per-window 头和 Retry-After → 以 per-window 头为准。
func TestHandle429_AnthropicPerWindowHeadersWinOverRetryAfter(t *testing.T) {
	repo := &sseRateLimitRepoStub{}
	svc := NewRateLimitService(repo, nil, &config.Config{}, nil, nil)

	reset5h := time.Now().Add(3 * time.Hour).Truncate(time.Second)
	headers := http.Header{}
	headers.Set("anthropic-ratelimit-unified-5h-reset", strconv.FormatInt(reset5h.Unix(), 10))
	headers.Set("anthropic-ratelimit-unified-5h-surpassed-threshold", "true")
	headers.Set("Retry-After", "120")

	svc.handle429(context.Background(), newMeridianAccountForSSETest(), headers, nil)

	require.Equal(t, 1, repo.rateLimitCalls)
	require.True(t, repo.lastRateLimitReset.Equal(reset5h), "got %v want %v", repo.lastRateLimitReset, reset5h)
	require.Equal(t, 1, repo.sessionWindowCalls)
}

func TestHandle429_AnthropicUnifiedResetWinsOverRetryAfter(t *testing.T) {
	repo := &sseRateLimitRepoStub{}
	svc := NewRateLimitService(repo, nil, &config.Config{}, nil, nil)

	reset := time.Now().Add(90 * time.Minute).Truncate(time.Second)
	headers := http.Header{}
	headers.Set("anthropic-ratelimit-unified-reset", strconv.FormatInt(reset.Unix(), 10))
	headers.Set("Retry-After", "120")

	svc.handle429(context.Background(), newMeridianAccountForSSETest(), headers, nil)

	require.Equal(t, 1, repo.rateLimitCalls)
	require.True(t, repo.lastRateLimitReset.Equal(reset), "got %v want %v", repo.lastRateLimitReset, reset)
}

// 场景 9：Retry-After 是过去时间 / 非法值 → 兜底；超过 8 天 → 按 8 天。
func TestHandle429_AnthropicRetryAfterInvalidOrOutOfRange(t *testing.T) {
	fallback := time.Duration(defaultRateLimit429CooldownSeconds) * time.Second
	eightDays := 8 * 24 * time.Hour

	cases := []struct {
		name       string
		retryAfter string
		want       time.Duration
	}{
		{name: "past http date", retryAfter: time.Now().Add(-time.Hour).UTC().Format(http.TimeFormat), want: fallback},
		{name: "zero seconds", retryAfter: "0", want: fallback},
		{name: "negative seconds", retryAfter: "-30", want: fallback},
		{name: "garbage", retryAfter: "soon", want: fallback},
		{name: "NaN", retryAfter: "NaN", want: fallback},
		{name: "Inf", retryAfter: "Inf", want: fallback},
		{name: "over 8 days", retryAfter: strconv.Itoa(30 * 24 * 3600), want: eightDays},
		{name: "huge seconds no overflow", retryAfter: "1e30", want: eightDays},
		{name: "future http date over 8 days", retryAfter: time.Now().Add(20 * 24 * time.Hour).UTC().Format(http.TimeFormat), want: eightDays},
		{name: "sub-second rounds up to min", retryAfter: "0.2", want: time.Second},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			repo := &sseRateLimitRepoStub{}
			svc := NewRateLimitService(repo, nil, &config.Config{}, nil, nil)

			before := time.Now()
			svc.handle429(context.Background(), newMeridianAccountForSSETest(), http.Header{"Retry-After": []string{tc.retryAfter}}, nil)
			after := time.Now()

			require.Equal(t, 1, repo.rateLimitCalls)
			requireResetAround(t, repo.lastRateLimitReset, before, after, tc.want)
		})
	}
}

func TestClampAnthropicRetryAfter(t *testing.T) {
	now := time.Now()
	_, ok := clampAnthropicRetryAfter(time.Time{}, now)
	require.False(t, ok)
	_, ok = clampAnthropicRetryAfter(now, now)
	require.False(t, ok)
	_, ok = clampAnthropicRetryAfter(now.Add(-time.Second), now)
	require.False(t, ok)

	got, ok := clampAnthropicRetryAfter(now.Add(100*time.Millisecond), now)
	require.True(t, ok)
	require.Equal(t, now.Add(time.Second), got)

	got, ok = clampAnthropicRetryAfter(now.Add(time.Hour), now)
	require.True(t, ok)
	require.Equal(t, now.Add(time.Hour), got)

	got, ok = clampAnthropicRetryAfter(now.Add(9*24*time.Hour), now)
	require.True(t, ok)
	require.Equal(t, now.Add(8*24*time.Hour), got)
}

// 场景 10：OpenAI 账号 → 行为不变（OpenAI 路径不经过 Anthropic Retry-After 分支）。
func TestHandle429_OpenAIBehaviorUnchanged(t *testing.T) {
	t.Run("no reset info uses fallback even with Retry-After", func(t *testing.T) {
		repo := &sseRateLimitRepoStub{}
		svc := NewRateLimitService(repo, nil, &config.Config{}, nil, nil)
		account := &Account{ID: 701, Platform: PlatformOpenAI, Type: AccountTypeAPIKey}

		before := time.Now()
		svc.handle429(context.Background(), account, http.Header{"Retry-After": []string{"120"}}, []byte(`{"error":{"message":"slow down"}}`))
		after := time.Now()

		require.Equal(t, 1, repo.rateLimitCalls)
		requireResetAround(t, repo.lastRateLimitReset, before, after, time.Duration(defaultRateLimit429CooldownSeconds)*time.Second)
	})

	t.Run("usage_limit_reached body still wins", func(t *testing.T) {
		repo := &sseRateLimitRepoStub{}
		svc := NewRateLimitService(repo, nil, &config.Config{}, nil, nil)
		account := &Account{ID: 702, Platform: PlatformOpenAI, Type: AccountTypeAPIKey}

		resetAt := time.Now().Add(2 * time.Hour).Truncate(time.Second)
		body := `{"error":{"type":"usage_limit_reached","resets_at":` + strconv.FormatInt(resetAt.Unix(), 10) + `}}`
		svc.handle429(context.Background(), account, http.Header{"Retry-After": []string{"120"}}, []byte(body))

		require.Equal(t, 1, repo.rateLimitCalls)
		require.True(t, repo.lastRateLimitReset.Equal(resetAt), "got %v want %v", repo.lastRateLimitReset, resetAt)
	})
}
