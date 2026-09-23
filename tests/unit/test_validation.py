from dataclasses import replace

import pytest

from adaptive_llm.contracts import Citation, ResponseFormat
from adaptive_llm.providers import FakeProvider, ProviderRequest
from adaptive_llm.validation import LocalValidator


async def test_supplied_citations_validate(provider_request: ProviderRequest) -> None:
    result = await FakeProvider().generate(provider_request)
    assert LocalValidator().validate(provider_request, result).passed
    invented = replace(result, citations=(Citation(document_id="invented", chunk_id="unknown"),))
    assert not LocalValidator().validate(provider_request, invented).passed
    inline = replace(result, content="SYNTHETIC [invented/unknown]")
    assert not LocalValidator().validate(provider_request, inline).passed
    assert not LocalValidator().validate(replace(provider_request, context=()), result).passed


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
async def test_empty_output_rejected(provider_request: ProviderRequest, content: str) -> None:
    result = replace(await FakeProvider().generate(provider_request), content=content)
    assert not LocalValidator().validate(provider_request, result).passed


@pytest.mark.parametrize(
    "content,valid",
    [
        ('{"synthetic": true}', True),
        ("{}", True),
        ("[]", False),
        ("null", False),
        ("{", False),
        ('{"value": NaN}', False),
    ],
)
async def test_json_object_validation(
    provider_request: ProviderRequest, content: str, valid: bool
) -> None:
    request = replace(provider_request, response_format=ResponseFormat(type="json_object"))
    result = replace(await FakeProvider().generate(request), content=content)
    assert LocalValidator().validate(request, result).passed is valid
