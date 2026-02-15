.PHONY: help train-all train-fno train-deeponet train-simple compare evaluate ood rollouts clean

help:
	@echo "Evaluation Framework - Publication-Grade Baselines"
	@echo ""
	@echo "Usage:"
	@echo "  make train-all      Train all baselines (~4-5 hours)"
	@echo "  make train-fno      Train FNO only (~2 hours)"
	@echo "  make train-deeponet Train DeepONet only (~2 hours)"
	@echo "  make train-simple   Train SimpleCNN only (~45 min)"
	@echo "  make evaluate       Run full evaluation (OOD + rollouts)"
	@echo "  make ood            Run OOD tests only"
	@echo "  make rollouts       Run long rollouts only"
	@echo "  make compare        Compare all models"
	@echo "  make clean          Remove checkpoints and results"

train-all: train-fno train-deeponet train-simple

train-fno:
	@echo "Training FNO baseline (this will take ~2 hours)..."
	python scripts/train_baseline.py --model fno --epochs 100 --batch-size 16

train-deeponet:
	@echo "Training DeepONet baseline (this will take ~2 hours)..."
	python scripts/train_baseline.py --model deeponet --epochs 100 --batch-size 16

train-simple:
	@echo "Training SimpleCNN baseline (this will take ~45 minutes)..."
	python scripts/train_baseline.py --model simple-cnn --epochs 50 --batch-size 16

evaluate:
	@echo "Running full evaluation (OOD tests + long rollouts)..."
	python scripts/run_full_evaluation.py --all

ood:
	@echo "Running OOD tests..."
	python scripts/run_full_evaluation.py --ood-only

rollouts:
	@echo "Running 100-step rollouts..."
	python scripts/run_full_evaluation.py --rollouts-only

compare:
	@echo "Comparing all models..."
	python scripts/compare_all.py \
		--test-data ../data/test \
		--checkpoints ../checkpoints \
		--output ../results/comparison.txt

clean:
	@echo "Cleaning checkpoints and results..."
	rm -f ../checkpoints/fno_best.pt
	rm -f ../checkpoints/deeponet_best.pt
	rm -f ../checkpoints/simple_baseline.pt
	rm -rf ../results/baselines/
