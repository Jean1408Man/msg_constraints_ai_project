from __future__ import annotations

import argparse
import time
from pathlib import Path

from .evaluator import MessageEvaluator
from .experiment import ExperimentRunner
from .llm import GroqConnectionError, GroqGenerator, build_generator
from .models import ExperimentRow, MessageInstance
from .simulator import InstanceSimulator
from .strategies import build_strategy
from .utils import read_jsonl, write_jsonl


def cmd_make_dataset(args: argparse.Namespace) -> None:
    sim = InstanceSimulator(seed=args.seed)
    instances = sim.generate_balanced(args.per_difficulty) if args.per_difficulty else sim.generate(args.n)
    write_jsonl(args.out, instances)
    print(f"Dataset generado: {args.out} ({len(instances)} instancias)")


def _build_cli_generator(args: argparse.Namespace):
    return build_generator(
        name=args.generator,
        seed=args.seed,
        model=args.model,
        groq_base_url=args.groq_base_url,
        temperature=args.temperature,
    )


def _build_cli_evaluator(args: argparse.Namespace) -> MessageEvaluator:
    return MessageEvaluator(
        open_constraint_evaluator=args.open_constraint_evaluator,
        semantic_evaluator=args.semantic_evaluator,
        model=args.model,
        groq_base_url=args.groq_base_url,
    )


def _progress_bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + "-" * width + "] 0/0 0.00%"
    ratio = min(1.0, max(0.0, done / total))
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total} {ratio:.2%}"


def _build_progress_callbacks(enabled: bool):
    if not enabled:
        return None, None, None

    started_at = time.perf_counter()

    def on_start(total: int, completed: int, incompatible: int) -> None:
        print(f"Progreso del experimento: {_progress_bar(completed, total)}", flush=True)
        if completed:
            print(f"Reanudando: {completed} combinaciones ya estaban completas.", flush=True)
        if incompatible:
            print(f"Se ignoraron {incompatible} filas incompatibles del CSV anterior.", flush=True)

    def on_task_start(completed: int, total: int, instance: MessageInstance, strategy: str) -> None:
        next_item = completed + 1
        print(
            f"Ejecutando {next_item}/{total}: "
            f"instancia={instance.instance_id} estrategia={strategy} dificultad={instance.difficulty}",
            flush=True,
        )

    def on_row_complete(completed: int, total: int, row: ExperimentRow) -> None:
        status = "ERROR" if row.error_type else ("OK" if row.valid else "FALLA")
        elapsed = time.perf_counter() - started_at
        error_suffix = f" error={row.error_type}" if row.error_type else ""
        print(
            f"{_progress_bar(completed, total)} "
            f"{row.instance_id}/{row.strategy} {status} "
            f"violaciones={row.violations} intentos={row.attempts} "
            f"fila={row.total_time_ms:.0f} ms acumulado={elapsed:.1f} s"
            f"{error_suffix}",
            flush=True,
        )

    return on_start, on_task_start, on_row_complete


def cmd_check_llm(args: argparse.Namespace) -> None:
    try:
        generator = GroqGenerator(model=args.model, base_url=args.groq_base_url, temperature=args.temperature)
        print("Probando conexión con Groq...")
        print(generator.healthcheck())
        print(f"OK: Groq respondió usando el modelo {generator.model}")
    except GroqConnectionError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)


def cmd_generate_one(args: argparse.Namespace) -> None:
    sim = InstanceSimulator(seed=args.seed)
    inst = sim.generate_one(1)
    try:
        evaluator = _build_cli_evaluator(args)
        generator = _build_cli_generator(args)
        strategy = build_strategy(
            args.strategy,
            generator,
            evaluator,
            max_attempts=args.max_attempts,
            population_size=args.population_size,
        )
        cand = strategy.run(inst)
    except GroqConnectionError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)

    ev = cand.evaluation or evaluator.evaluate(cand.text, inst)

    print("INSTANCIA")
    print(inst.to_json())
    print("\nMENSAJE")
    print(cand.text)
    print("\nEVALUACIÓN")
    model_info = getattr(generator, "model", "no aplica")
    print(f"generador={args.generator} modelo={model_info}")
    print(f"evaluador_restricciones_abiertas={ev.open_constraint_evaluator}")
    print(f"evaluador_semántico={ev.semantic_evaluator}")
    print(f"válido={ev.valid} hard_score={ev.hard_score} semantic_score={ev.semantic_score} total={ev.total_score}")
    for r in ev.constraint_results:
        missing = f"; faltante={', '.join(r.missing)}" if r.missing else ""
        print(f"- {r.name}: {'OK' if r.passed else 'FALLA'} [{r.source}] ({r.details}{missing})")
    if ev.notes:
        print("Notas:", "; ".join(ev.notes))


def cmd_run_experiment(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"No existe {dataset_path}. Generando dataset de ejemplo...")
        instances = InstanceSimulator(seed=args.seed).generate(args.n)
        write_jsonl(dataset_path, instances)
    else:
        instances = read_jsonl(dataset_path)

    runner = ExperimentRunner(
        seed=args.seed,
        generator_name=args.generator,
        model=args.model,
        groq_base_url=args.groq_base_url,
        temperature=args.temperature,
        open_constraint_evaluator=args.open_constraint_evaluator,
        semantic_evaluator=args.semantic_evaluator,
        max_attempts=args.max_attempts,
        population_size=args.population_size,
    )
    on_start, on_task_start, on_row_complete = _build_progress_callbacks(not args.no_progress)
    try:
        rows = runner.run_to_csv(
            args.out,
            instances,
            strategies=args.strategies,
            resume=args.resume,
            on_start=on_start,
            on_task_start=on_task_start,
            on_row_complete=on_row_complete,
        )
    except GroqConnectionError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
    print(ExperimentRunner.summarize(rows))
    print(f"Resultados guardados en: {args.out}")


def add_generator_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--generator",
        choices=["groq"],
        default="groq",
        help="Generador LLM mediante API de Groq",
    )
    parser.add_argument("--model", default=None, help="Modelo de Groq; si se omite usa GROQ_MODEL")
    parser.add_argument("--groq-base-url", default=None, help="URL base compatible OpenAI; si se omite usa GROQ_BASE_URL o Groq")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperatura del LLM")
    parser.add_argument(
        "--semantic-evaluator",
        choices=["heuristic", "llm"],
        default="llm",
        help="heuristic = rúbrica local; llm = rúbrica semántica evaluada por Groq",
    )
    parser.add_argument(
        "--open-constraint-evaluator",
        choices=["heuristic", "llm"],
        default="llm",
        help="llm = Groq juzga saludo/cuerpo/cierre/agradecimiento; heuristic = respaldo local aproximado",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generador de mensajes bajo restricciones")
    sub = parser.add_subparsers(required=True)

    p1 = sub.add_parser("make-dataset", help="Genera dataset simulado JSONL")
    p1.add_argument("--n", type=int, default=50)
    p1.add_argument("--per-difficulty", type=int, default=None, help="Instancias por dificultad: baja, media y alta")
    p1.add_argument("--out", default="data/instances.jsonl")
    p1.add_argument("--seed", type=int, default=42)
    p1.set_defaults(func=cmd_make_dataset)

    p0 = sub.add_parser("check-llm", help="Verifica que Groq responda")
    add_generator_args(p0)
    p0.set_defaults(func=cmd_check_llm)

    p2 = sub.add_parser("generate-one", help="Genera y evalúa una instancia")
    p2.add_argument("--strategy", choices=["direct", "verify", "repair", "multi"], default="repair")
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--max-attempts", type=int, default=3, help="Intentos máximos para repair; presupuesto total en multi")
    p2.add_argument("--population-size", type=int, default=2, help="Candidatos iniciales para multi")
    add_generator_args(p2)
    p2.set_defaults(func=cmd_generate_one)

    p3 = sub.add_parser("run-experiment", help="Ejecuta comparación experimental")
    p3.add_argument("--dataset", default="data/instances.jsonl")
    p3.add_argument("--out", default="results/experiment_results.csv")
    p3.add_argument("--n", type=int, default=50, help="Tamaño si hay que crear dataset")
    p3.add_argument("--seed", type=int, default=42)
    p3.add_argument("--max-attempts", type=int, default=3, help="Intentos máximos para repair; presupuesto total en multi")
    p3.add_argument("--population-size", type=int, default=2, help="Candidatos iniciales para multi")
    p3.add_argument("--resume", action="store_true", help="No repite pares instance_id + strategy ya escritos en el CSV")
    p3.add_argument("--no-progress", action="store_true", help="Desactiva la salida de progreso durante el experimento")
    p3.add_argument(
        "--strategies",
        nargs="+",
        choices=["direct", "verify", "repair", "multi"],
        default=None,
        help="Estrategias a comparar; si se omite usa direct, repair y multi",
    )
    add_generator_args(p3)
    p3.set_defaults(func=cmd_run_experiment)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
