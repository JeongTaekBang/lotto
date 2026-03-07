#!/usr/bin/env python
"""
로또 예측 시스템 - 멀티 모델 아키텍처

사용법:
    python main_new.py train [--model=MODEL] [--epochs=N]
    python main_new.py predict [--model=MODEL] [--ensemble] [--sets=N]
    python main_new.py evaluate [--model=MODEL] [--rounds=N]
    python main_new.py compare [--rounds=N]
    python main_new.py crawl
    python main_new.py analyze
"""
import sys
import os
import argparse

# 경로 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasources.mysql_source import MySQLDataSource
from models.factory import ModelFactory
from training.trainer import UnifiedTrainer
from training.evaluator import ModelEvaluator
from ensemble.manager import EnsembleManager
from ensemble.weighted_average import WeightedAverageEnsemble
from ensemble.voting import VotingEnsemble
from filters.pattern_filter import PatternFilter
from filters.statistical_filter import StatisticalFilter
from filters.frequency_filter import FrequencyFilter
from filters.composite_filter import CompositeFilter


def get_feature_mode(args) -> str:
    """args에서 feature_mode 결정 (하위 호환성 유지)"""
    if hasattr(args, 'feature_mode') and args.feature_mode != 'basic':
        return args.feature_mode
    if hasattr(args, 'extended') and args.extended:
        return 'extended'
    return 'basic'


def cmd_train(args):
    """모델 학습"""
    print("=" * 60)
    print("로또 예측 모델 학습")
    print("=" * 60)

    # 데이터 소스
    datasource = MySQLDataSource()

    # 모델 및 피처 모드 결정
    models_to_train = []
    if args.model == 'all':
        # Train All: Extended 고정, XGBoost 제외
        models_to_train = [m for m in ModelFactory.list_models() if m != 'xgboost']
        feature_mode = 'extended'
        print("\n[Train All] Extended 피처 모드, XGBoost 제외")
    else:
        models_to_train = [args.model]
        # 단일 모델: XGBoost는 Basic, 나머지는 Extended
        if args.model == 'xgboost':
            feature_mode = 'basic'
        else:
            feature_mode = 'extended'

    # 트레이너 설정
    trainer = UnifiedTrainer(
        datasource,
        seq_length=args.seq_length,
        feature_mode=feature_mode,
        use_bonus=args.bonus
    )

    feature_dims = {'basic': 45, 'extended': 74}
    print(f"\n학습할 모델: {models_to_train}")
    print(f"에포크: {args.epochs}")
    print(f"피처 모드: {feature_mode} ({feature_dims[feature_mode]}차원)")
    print("-" * 60)

    for model_name in models_to_train:
        try:
            print(f"\n>>> {model_name} 학습 시작")
            model, history = trainer.create_and_train(
                model_name,
                epochs=args.epochs,
                batch_size=args.batch_size
            )

            # 모델 저장
            save_path = f"saved_models/{model_name}.pt"
            os.makedirs("saved_models", exist_ok=True)

            # 모델 타입에 따라 저장 방식 결정
            if model_name in ['xgboost', 'random_forest', 'markov']:
                save_path = f"saved_models/{model_name}.pkl"

            model.save(save_path)
            print(f"모델 저장: {save_path}")

        except Exception as e:
            print(f"[오류] {model_name} 학습 실패: {e}")

    print("\n" + "=" * 60)
    print("학습 완료!")


def cmd_predict(args):
    """번호 예측"""
    print("=" * 60)
    print("로또 번호 예측")
    print("=" * 60)

    datasource = MySQLDataSource()

    # 피처 모드 자동 결정
    if args.ensemble:
        # 앙상블: Extended 고정
        feature_mode = 'extended'
    else:
        # 단일 모델: XGBoost는 Basic, 나머지는 Extended
        feature_mode = 'basic' if args.model == 'xgboost' else 'extended'

    trainer = UnifiedTrainer(
        datasource,
        seq_length=args.seq_length,
        feature_mode=feature_mode,
        use_bonus=args.bonus
    )

    # 최신 시퀀스
    latest_seq = trainer.get_latest_sequence()
    last_round = datasource.get_last_round_num()

    if args.ensemble:
        # 앙상블 예측 (Extended 고정, XGBoost 제외)
        manager = EnsembleManager()

        # 모델 로드 (XGBoost 제외)
        available_models = []
        for model_name in ModelFactory.list_models():
            if model_name == 'xgboost':
                continue  # 앙상블에서 XGBoost 제외
            try:
                ext = '.pkl' if model_name in ['random_forest', 'markov'] else '.pt'
                path = f"saved_models/{model_name}{ext}"
                if os.path.exists(path):
                    config = {'input_dim': trainer.feature_dim, 'seq_length': trainer.seq_length}
                    model = ModelFactory.create(model_name, config)
                    model.load(path)
                    manager.register_model(model_name, model)
                    available_models.append(model_name)
            except Exception as e:
                print(f"[경고] {model_name} 로드 실패: {e}")

        if not available_models:
            print("로드된 모델이 없습니다. 먼저 학습을 진행해주세요.")
            return

        print(f"앙상블 모델: {available_models}")

        # 앙상블 전략 설정
        if args.strategy == 'voting':
            manager.set_strategy(VotingEnsemble())
        else:
            manager.set_strategy(WeightedAverageEnsemble())

        # 자동 가중치 계산 (성능 기반)
        if args.auto_weight:
            print("\n성능 기반 자동 가중치 계산 중...")
            X, y = trainer.prepare_sequences()
            # 최근 100회차로 평가
            eval_size = min(100, len(X))
            X_eval, y_eval = X[-eval_size:], y[-eval_size:]
            weights = manager.auto_weight_by_performance(
                X_eval, y_eval,
                method=args.weight_method,
                verbose=True
            )

        # 필터 설정 (mode 사용 시에도 스코어링에 활용됨)
        if args.filter or args.mode:
            composite = CompositeFilter()
            composite.add_filter(PatternFilter())
            composite.add_filter(StatisticalFilter())
            composite.add_filter(FrequencyFilter(datasource))
            manager.add_filter(composite)

        predictions = manager.predict(
            latest_seq, num_sets=args.sets, mode=args.mode
        )

    else:
        # 단일 모델 예측
        model_name = args.model
        ext = '.pkl' if model_name in ['xgboost', 'random_forest', 'markov'] else '.pt'
        path = f"saved_models/{model_name}{ext}"

        if not os.path.exists(path):
            print(f"모델 파일이 없습니다: {path}")
            print("먼저 학습을 진행해주세요: python main_new.py train")
            return

        config = {'input_dim': trainer.feature_dim, 'seq_length': trainer.seq_length}
        model = ModelFactory.create(model_name, config)
        model.load(path)

        if args.mode:
            from core.selector import CandidateSelector
            proba = model.predict_proba(latest_seq)
            filters = []
            if args.filter or args.mode:
                filters = [PatternFilter(), StatisticalFilter(), FrequencyFilter(datasource)]
            selector = CandidateSelector(filters=filters, preset=args.mode)
            predictions = selector.select(proba, num_sets=args.sets, model_name=model_name)
        else:
            predictions = model.predict_numbers(latest_seq, num_sets=args.sets)

    # 결과 출력
    mode_label = f", 모드: {args.mode}" if args.mode else ""
    print(f"\n=== {last_round + 1}회차 예측 ({args.sets}세트{mode_label}) ===\n")
    for i, pred in enumerate(predictions, 1):
        score_info = ""
        if 'score' in pred.metadata:
            score_info = f"  점수: {pred.metadata['score']:.2f}"
        print(f"  {i}번: {pred.numbers}  (신뢰도: {pred.confidence:.3f}{score_info})")

    if args.ensemble:
        weight_info = "자동" if args.auto_weight else "균등"
        print(f"\n앙상블: {args.strategy}, 가중치: {weight_info}, 필터: {args.filter or bool(args.mode)}")


def cmd_evaluate(args):
    """모델 평가 (백테스팅)"""
    print("=" * 60)
    print("모델 평가 (백테스팅)")
    print("=" * 60)

    datasource = MySQLDataSource()

    # 모델별 피처 모드 결정 (학습 시와 동일하게)
    # XGBoost: basic (45차원), 나머지: extended (74차원)
    models_to_eval = []
    if args.model == 'all':
        models_to_eval = ModelFactory.list_models()
    else:
        models_to_eval = [args.model]

    evaluator = ModelEvaluator()

    # 피처 모드별로 분리하여 평가
    # Extended 모델들 (XGBoost 제외)
    extended_models = [m for m in models_to_eval if m != 'xgboost']
    # Basic 모델 (XGBoost)
    basic_models = [m for m in models_to_eval if m == 'xgboost']

    # Extended 모델 평가
    if extended_models:
        trainer_ext = UnifiedTrainer(
            datasource,
            seq_length=args.seq_length,
            feature_mode='extended',
            use_bonus=args.bonus
        )
        X_ext, y_ext = trainer_ext.prepare_sequences()
        if args.rounds < len(X_ext):
            X_ext = X_ext[-args.rounds:]
            y_ext = y_ext[-args.rounds:]

        for model_name in extended_models:
            try:
                ext = '.pkl' if model_name in ['random_forest', 'markov'] else '.pt'
                path = f"saved_models/{model_name}{ext}"

                if not os.path.exists(path):
                    print(f"[건너뜀] {model_name}: 모델 파일 없음")
                    continue

                config = {'input_dim': trainer_ext.feature_dim, 'seq_length': trainer_ext.seq_length}
                model = ModelFactory.create(model_name, config)
                model.load(path)

                print(f"\n>>> {model_name} 평가 중... (extended, {trainer_ext.feature_dim}차원)")
                result = evaluator.evaluate(model, X_ext, y_ext, model_name)
                print(result.summary())

            except Exception as e:
                print(f"[오류] {model_name} 평가 실패: {e}")

    # Basic 모델 평가 (XGBoost)
    if basic_models:
        trainer_basic = UnifiedTrainer(
            datasource,
            seq_length=args.seq_length,
            feature_mode='basic',
            use_bonus=args.bonus
        )
        X_basic, y_basic = trainer_basic.prepare_sequences()
        if args.rounds < len(X_basic):
            X_basic = X_basic[-args.rounds:]
            y_basic = y_basic[-args.rounds:]

        for model_name in basic_models:
            try:
                path = f"saved_models/{model_name}.pkl"

                if not os.path.exists(path):
                    print(f"[건너뜀] {model_name}: 모델 파일 없음")
                    continue

                config = {'input_dim': trainer_basic.feature_dim, 'seq_length': trainer_basic.seq_length}
                model = ModelFactory.create(model_name, config)
                model.load(path)

                print(f"\n>>> {model_name} 평가 중... (basic, {trainer_basic.feature_dim}차원)")
                result = evaluator.evaluate(model, X_basic, y_basic, model_name)
                print(result.summary())

            except Exception as e:
                print(f"[오류] {model_name} 평가 실패: {e}")

    # 모델 비교
    if len(evaluator.results) > 1:
        evaluator.compare_models()


def cmd_compare(args):
    """모델 비교"""
    print("=" * 60)
    print("모델 성능 비교")
    print("=" * 60)

    # evaluate와 동일하지만 모든 모델 대상
    args.model = 'all'
    cmd_evaluate(args)


def cmd_crawl(args):
    """최신 데이터 크롤링"""
    print("=" * 60)
    print("최신 당첨번호 크롤링")
    print("=" * 60)

    try:
        from crawling import LottoCrawler
        crawler = LottoCrawler()
        crawler.update_latest()
    except ImportError:
        print("crawling.py 모듈을 찾을 수 없습니다.")


def cmd_analyze(args):
    """통계 분석"""
    print("=" * 60)
    print("통계 분석")
    print("=" * 60)

    try:
        from analysis.statistics_report import LottoStatistics
        stats = LottoStatistics()
        stats.run_full_analysis()
    except ImportError:
        print("analysis/statistics_report.py 모듈을 찾을 수 없습니다.")


def cmd_list(args):
    """사용 가능한 모델 목록"""
    print("사용 가능한 모델:")
    for name in ModelFactory.list_models():
        print(f"  - {name}")


def main():
    parser = argparse.ArgumentParser(
        description="로또 예측 시스템 - 멀티 모델 아키텍처",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='명령어')

    # train
    train_parser = subparsers.add_parser('train', help='모델 학습')
    train_parser.add_argument('--model', type=str, default='transformer',
                             help='학습할 모델 (transformer, lstm, xgboost, ... 또는 all)')
    train_parser.add_argument('--epochs', type=int, default=100, help='에포크 수')
    train_parser.add_argument('--batch-size', type=int, default=32, help='배치 크기')
    train_parser.add_argument('--seq-length', type=int, default=20, help='시퀀스 길이')
    train_parser.add_argument('--feature-mode', type=str, default='basic',
                             choices=['basic', 'extended', 'xgboost'],
                             help='피처 모드: basic(45), extended(74), xgboost(66)')
    train_parser.add_argument('--extended', action='store_true', help='(deprecated) --feature-mode=extended')
    train_parser.add_argument('--bonus', action='store_true', help='보너스 번호 피처 사용')

    # predict
    predict_parser = subparsers.add_parser('predict', help='번호 예측')
    predict_parser.add_argument('--model', type=str, default='transformer', help='사용할 모델')
    predict_parser.add_argument('--sets', type=int, default=5, help='생성할 세트 수')
    predict_parser.add_argument('--ensemble', action='store_true', help='앙상블 사용')
    predict_parser.add_argument('--strategy', type=str, default='weighted',
                               choices=['weighted', 'voting'], help='앙상블 전략')
    predict_parser.add_argument('--auto-weight', action='store_true',
                               help='성능 기반 자동 가중치 (기본값: True)')
    predict_parser.add_argument('--weight-method', type=str, default='softmax',
                               choices=['softmax', 'linear', 'rank'],
                               help='가중치 계산 방식 (softmax/linear/rank)')
    predict_parser.add_argument('--filter', action='store_true', help='필터 적용')
    predict_parser.add_argument('--mode', type=str, default=None,
                               choices=['safe', 'balanced', 'aggressive'],
                               help='추출 모드: safe(보수적), balanced(균형), aggressive(공격적)')
    predict_parser.add_argument('--seq-length', type=int, default=20)
    predict_parser.add_argument('--feature-mode', type=str, default='basic',
                               choices=['basic', 'extended', 'xgboost'],
                               help='피처 모드: basic(45), extended(74), xgboost(66)')
    predict_parser.add_argument('--extended', action='store_true', help='(deprecated)')
    predict_parser.add_argument('--bonus', action='store_true')

    # evaluate
    eval_parser = subparsers.add_parser('evaluate', help='모델 평가')
    eval_parser.add_argument('--model', type=str, default='transformer', help='평가할 모델')
    eval_parser.add_argument('--rounds', type=int, default=100, help='평가할 회차 수')
    eval_parser.add_argument('--seq-length', type=int, default=20)
    eval_parser.add_argument('--feature-mode', type=str, default='basic',
                            choices=['basic', 'extended', 'xgboost'],
                            help='피처 모드: basic(45), extended(74), xgboost(66)')
    eval_parser.add_argument('--extended', action='store_true', help='(deprecated)')
    eval_parser.add_argument('--bonus', action='store_true')

    # compare
    compare_parser = subparsers.add_parser('compare', help='모델 비교')
    compare_parser.add_argument('--rounds', type=int, default=100)
    compare_parser.add_argument('--seq-length', type=int, default=20)
    compare_parser.add_argument('--feature-mode', type=str, default='basic',
                               choices=['basic', 'extended', 'xgboost'],
                               help='피처 모드: basic(45), extended(74), xgboost(66)')
    compare_parser.add_argument('--extended', action='store_true', help='(deprecated)')
    compare_parser.add_argument('--bonus', action='store_true')

    # crawl
    subparsers.add_parser('crawl', help='최신 데이터 크롤링')

    # analyze
    subparsers.add_parser('analyze', help='통계 분석')

    # list
    subparsers.add_parser('list', help='사용 가능한 모델 목록')

    args = parser.parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'predict':
        cmd_predict(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'crawl':
        cmd_crawl(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'list':
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
