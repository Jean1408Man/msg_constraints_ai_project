from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from messagegen.experiment import ExperimentRunner
from messagegen.simulator import InstanceSimulator
from messagegen.utils import write_jsonl


def main():
    instances = InstanceSimulator(seed=123).generate(3)
    write_jsonl(ROOT / "data" / "sample_instances.jsonl", instances)
    runner = ExperimentRunner(seed=123, generator_name="groq", max_attempts=3, population_size=2)
    rows = runner.run_to_csv(ROOT / "results" / "experiment_results.csv", instances, strategies=["repair"])
    print(runner.summarize(rows))
    print("Archivos creados en data/ y results/.")


if __name__ == "__main__":
    main()
