from fastapi.testclient import TestClient

from mini_llm.api import GenerateRequest, GenerateResponse, create_app


class FakeGenerator:
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        return GenerateResponse(
            generated_text=f"{request.prompt}の続き",
            generated_token_count=3,
            checkpoint_step=20,
        )


def test_reports_health_after_model_is_loaded() -> None:
    with TestClient(create_app(service=FakeGenerator())) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_generates_text_from_validated_request() -> None:
    with TestClient(create_app(service=FakeGenerator())) as client:
        response = client.post(
            "/api/generate",
            json={
                "prompt": "Pythonで",
                "max_new_tokens": 32,
                "temperature": 0.7,
                "top_k": 20,
                "seed": 7,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "generated_text": "Pythonでの続き",
        "generated_token_count": 3,
        "checkpoint_step": 20,
    }


def test_rejects_empty_prompt_and_excessive_generation_length() -> None:
    with TestClient(create_app(service=FakeGenerator())) as client:
        response = client.post(
            "/api/generate",
            json={"prompt": " ", "max_new_tokens": 1000},
        )

    assert response.status_code == 422
