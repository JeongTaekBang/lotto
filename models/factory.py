"""모델 팩토리 - 다양한 모델 생성"""
from typing import Dict, Type, List, Any
from core.base_model import BaseModel


class ModelFactory:
    """모델 팩토리 - 플러그인 방식으로 모델 생성"""

    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """
        모델 클래스 등록

        Args:
            name: 모델 이름
            model_class: BaseModel을 상속한 클래스
        """
        cls._registry[name] = model_class

    @classmethod
    def create(cls, name: str, config: Dict[str, Any] = None) -> BaseModel:
        """
        모델 생성

        Args:
            name: 모델 이름
            config: 모델 설정

        Returns:
            생성된 모델 인스턴스
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._registry.keys())}")

        return cls._registry[name](config or {})

    @classmethod
    def list_models(cls) -> List[str]:
        """등록된 모델 목록"""
        return list(cls._registry.keys())

    @classmethod
    def get_class(cls, name: str) -> Type[BaseModel]:
        """모델 클래스 반환"""
        if name not in cls._registry:
            raise ValueError(f"Unknown model: {name}")
        return cls._registry[name]


# 기본 모델 등록
def _register_default_models():
    """기본 모델들 등록"""
    try:
        from .transformer import TransformerModel
        ModelFactory.register('transformer', TransformerModel)
    except ImportError:
        pass

    try:
        from .lstm import LSTMModel
        ModelFactory.register('lstm', LSTMModel)
    except ImportError:
        pass

    try:
        from .gru import GRUModel
        ModelFactory.register('gru', GRUModel)
    except ImportError:
        pass

    try:
        from .xgboost_model import XGBoostModel
        ModelFactory.register('xgboost', XGBoostModel)
    except ImportError:
        pass

    try:
        from .random_forest import RandomForestModel
        ModelFactory.register('random_forest', RandomForestModel)
    except ImportError:
        pass

    try:
        from .markov import MarkovChainModel
        ModelFactory.register('markov', MarkovChainModel)
    except ImportError:
        pass

    try:
        from .cnn_grid import CNNGridModel
        ModelFactory.register('cnn_grid', CNNGridModel)
    except ImportError:
        pass


_register_default_models()


if __name__ == "__main__":
    print("등록된 모델:", ModelFactory.list_models())

    # Transformer 생성 테스트
    model = ModelFactory.create('transformer', {'input_dim': 45})
    print(f"생성된 모델: {model}")
