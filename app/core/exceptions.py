"""模型供应商异常体系。

所有异常消息必须满足：
- 中文可读、带排查建议；
- 不携带 API Key 等敏感信息（调用方在记录日志时也应避免打印请求头）。
"""

from __future__ import annotations


class ModelProviderError(Exception):
    """所有模型调用异常的基类。``retryable`` 表示是否值得重试。"""

    retryable: bool = False

    def __init__(self, message: str, *, provider: str = "", model: str = "") -> None:
        self.provider = provider
        self.model = model
        super().__init__(message)


class AuthenticationError(ModelProviderError):
    """401：API Key 无效或过期。不可重试。"""

    retryable = False


class ModelNotFoundError(ModelProviderError):
    """404：模型名不存在或账号无权限。不可重试。"""

    retryable = False


class RateLimitError(ModelProviderError):
    """429：触发限流或余额不足。可重试（退避）。"""

    retryable = True


class UpstreamTimeoutError(ModelProviderError):
    """上游超时。可重试。"""

    retryable = True


class UpstreamServerError(ModelProviderError):
    """5xx：上游服务暂时不可用。可重试。"""

    retryable = True


class BadRequestError(ModelProviderError):
    """400：请求参数非法（如超长、格式错）。不可重试。"""

    retryable = False


class RetryExhaustedError(ModelProviderError):
    """重试次数耗尽后的包装异常。"""

    retryable = False
