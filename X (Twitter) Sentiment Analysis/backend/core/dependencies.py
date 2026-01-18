from core.model_loader import get_model_service,ModelService

def model_service_dep()->ModelService:
    return get_model_service()