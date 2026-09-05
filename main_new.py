#!/usr/bin/env python
"""
로또 예측 시스템 - 멀티 모델 아키텍처

사용법:
    python main_new.py train [--model=MODEL] [--epochs=N]
    python main_new.py predict [--model=MODEL] [--ensemble] [--sets=N]
    python main_new.py evaluate [--model=MODEL] [--rounds=N]
    python main_new.py compare [--rounds=N]
    python main_new.py crawl [--no-pull]
    python main_new.py analyze
"""
import sys
import os
import argparse
import subprocess

# 경로 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasources.sqlite_source import SQLiteDataSource
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


def _make_datasource(args) -> SQLiteDataSource:
    """CLI 인자로부터 데이터 소스 생성 (--no-fetch 시 API 보충 비활성화)"""
    return SQLiteDataSource(fetch_missing=not getattr(args, 'no_fetch', False))


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
    datasource = _make_datasource(args)

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

    datasource = _make_datasource(args)

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

        # 자동 가중치 계산 (성능 기반, 테스트셋만 사용)
        if args.auto_weight:
            print("\n성능 기반 자동 가중치 계산 중...")
            X_eval, y_eval = _get_test_data(trainer, 100)
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


def _get_test_data(trainer, args_rounds, test_ratio=0.1):
    """학습에 사용되지 않은 테스트 데이터만 반환 (데이터 누수 방지)"""
    X, y = trainer.prepare_sequences()
    split_idx = int(len(X) * (1 - test_ratio))
    X_test, y_test = X[split_idx:], y[split_idx:]

    if len(X_test) == 0:
        raise ValueError("테스트 데이터가 비어 있습니다. 학습 데이터가 충분한지 확인하세요.")

    # 요청 라운드가 유효한 양수이고 테스트셋보다 작으면 뒤에서 잘라냄
    if 0 < args_rounds < len(X_test):
        X_test = X_test[-args_rounds:]
        y_test = y_test[-args_rounds:]

    return X_test, y_test


def cmd_evaluate(args):
    """모델 평가 (백테스팅) - 학습 데이터 제외, 테스트셋만 사용"""
    print("=" * 60)
    print("모델 평가 (백테스팅)")
    print("=" * 60)

    datasource = _make_datasource(args)

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
        X_ext, y_ext = _get_test_data(trainer_ext, args.rounds)
        print(f"\n[평가] 테스트셋: {len(X_ext)}회차 (학습 데이터 제외)")

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
        X_basic, y_basic = _get_test_data(trainer_basic, args.rounds)
        print(f"\n[평가] 테스트셋: {len(X_basic)}회차 (학습 데이터 제외)")

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


def _pull_blog_repo(db_path):
    """블로그 저장소에서 git pull --ff-only 시도 (실패해도 종료 코드는 정상)"""
    repo_dir = db_path.parent.parent  # <repo>/data/lotto.db -> <repo>
    print(f"\n[블로그 저장소 갱신] {repo_dir}")

    if not (repo_dir / '.git').exists():
        print("  git 저장소가 아니어서 건너뜁니다.")
        return

    try:
        result = subprocess.run(
            ['git', '-C', str(repo_dir), 'pull', '--ff-only'],
            capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"  git pull 실행 실패: {e}")
        return

    for line in (result.stdout + result.stderr).strip().splitlines():
        print(f"  {line}")

    if result.returncode != 0:
        print(f"  git pull --ff-only 실패 (exit {result.returncode}) — 기존 DB 파일을 그대로 사용합니다.")


def cmd_crawl(args):
    """당첨번호 데이터 상태 확인 (DB·API 회차 차이 보고 + 블로그 저장소 갱신)"""
    print("=" * 60)
    print("당첨번호 데이터 상태 확인")
    print("=" * 60)

    # 상태 확인 단계에서는 API 보충 없이 DB 원본만 읽는다
    source = SQLiteDataSource(fetch_missing=False)
    print(f"DB 경로: {source.db_path}")

    try:
        records = source.load()
    except (FileNotFoundError, ValueError) as e:
        print(f"[오류] {e}")
        return

    db_last = records[-1].round_num if records else 0
    api_last = source.fetch_latest_round()

    print(f"\nDB 마지막 회차: {db_last}")
    if api_last is None:
        print("동행복권 최신 회차: 확인 실패")
    else:
        print(f"동행복권 최신 회차: {api_last}")
        gap = api_last - db_last
        if gap > 0:
            print(f"  -> DB에 {gap}회차 부족 ({db_last + 1}~{api_last}). "
                  f"예측 시 API로 메모리 보충됩니다 (--no-fetch로 비활성화).")
        else:
            print("  -> DB가 최신 상태입니다.")

    if args.no_pull:
        print("\n[건너뜀] --no-pull 지정으로 블로그 저장소 갱신을 생략합니다.")
        return

    _pull_blog_repo(source.db_path)


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


def _positive_int(value):
    """argparse용 양의 정수 검증"""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"양의 정수를 입력하세요 (입력값: {value})")
    return ivalue


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
    train_parser.add_argument('--no-fetch', action='store_true',
                             help='DB보다 최신인 회차를 API로 보충하지 않음')

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
    predict_parser.add_argument('--no-fetch', action='store_true',
                               help='DB보다 최신인 회차를 API로 보충하지 않음')

    # evaluate
    eval_parser = subparsers.add_parser('evaluate', help='모델 평가')
    eval_parser.add_argument('--model', type=str, default='transformer', help='평가할 모델')
    eval_parser.add_argument('--rounds', type=_positive_int, default=100, help='평가할 회차 수 (양수)')
    eval_parser.add_argument('--seq-length', type=int, default=20)
    eval_parser.add_argument('--feature-mode', type=str, default='basic',
                            choices=['basic', 'extended', 'xgboost'],
                            help='피처 모드: basic(45), extended(74), xgboost(66)')
    eval_parser.add_argument('--extended', action='store_true', help='(deprecated)')
    eval_parser.add_argument('--bonus', action='store_true')
    eval_parser.add_argument('--no-fetch', action='store_true',
                            help='DB보다 최신인 회차를 API로 보충하지 않음')

    # compare
    compare_parser = subparsers.add_parser('compare', help='모델 비교')
    compare_parser.add_argument('--rounds', type=_positive_int, default=100)
    compare_parser.add_argument('--seq-length', type=int, default=20)
    compare_parser.add_argument('--feature-mode', type=str, default='basic',
                               choices=['basic', 'extended', 'xgboost'],
                               help='피처 모드: basic(45), extended(74), xgboost(66)')
    compare_parser.add_argument('--extended', action='store_true', help='(deprecated)')
    compare_parser.add_argument('--bonus', action='store_true')
    compare_parser.add_argument('--no-fetch', action='store_true',
                               help='DB보다 최신인 회차를 API로 보충하지 않음')

    # crawl
    crawl_parser = subparsers.add_parser('crawl', help='데이터 상태 확인 및 블로그 저장소 갱신')
    crawl_parser.add_argument('--no-pull', action='store_true',
                              help='블로그 저장소 git pull --ff-only 생략')

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
