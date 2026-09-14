import os

from test_sets import get_skript_data, get_skript_formula_data, get_slides_data
from chunking import ChunkingMethod
import database as db

# Leere-Retrieval-Hinweise (ZeroDivision) nur bei DEBUG – sonst deckt der
# Coverage-Report im __main__ dieselbe Information sauber ab.
DEBUG = os.getenv("DEBUG", "false").strip().lower() in ("1", "true", "yes", "ja")

SKRIPT_TEST = get_skript_data()
SKRIPT_FORMULA_TEST = get_skript_formula_data()
SLIDES_TEST = get_slides_data()

tests = {
    "SKRIPT_TEST": SKRIPT_TEST,
    "SKRIPT_FORMULA_TEST": SKRIPT_FORMULA_TEST,
    "SLIDES_TEST": SLIDES_TEST,
}



def run_evaluation(dynamic: bool, threshold: float = 0.7, n_param: int = 3):
    results = {}
    specific_results = []
    for method in ChunkingMethod:
        method_results = {}
        method_empty = 0  # Fragen ohne einen einzigen abgerufenen Chunk (coverage-Report)
        for test_name, test in tests.items():
            test_results = {}
            acc_pr: float = 0.0
            acc_rc: float = 0.0
            acc_iou: float = 0.0
            acc_f1: float = 0.0
            for test_case in test:
                if dynamic:
                    retrieval = db.retrieve_chunks_dynamic(query=test_case['query'], thresh_hold=threshold, method=method.value)
                else:
                    retrieval = db.retrieve_chunks(query=test_case['query'], n=n_param, method=method.value)
                retrieval = "\n".join(retrieval["documents"][0])
                is_empty = not retrieval.strip()  # nichts abgerufen (z. B. Schwelle zu streng)
                if is_empty:
                    method_empty += 1
                pr, rc, iou, f1 = calculate_iou(ground_truth=string_to_tokens(test_case['ground_truth']),
                                            retrieved=string_to_tokens(retrieval))
                acc_pr += pr
                acc_rc += rc
                acc_iou += iou
                acc_f1 += f1
                result = {
                    'id': test_case['id'],
                    'precision': pr,
                    'recall': rc,
                    'iou': iou,
                    'f1': f1,
                    'empty': is_empty,
                    'method': method.value,
                }
                specific_results.append(result)
            n = len(test)
            if n != 0:
                avg_pr = acc_pr / n
                avg_rc = acc_rc / n
                avg_iou = acc_iou / n
                avg_f1 = acc_f1 / n
                test_results["precision"] = avg_pr
                test_results["recall"] = avg_rc
                test_results["iou"] = avg_iou
                test_results["f1"] = avg_f1
                method_results[test_name] = test_results
            else:
                test_results["precision"] = 0.0
                test_results["recall"] = 0.0
                test_results["iou"] = 0.0
                test_results["f1"] = 0.0
                method_results[test_name] = test_results

        all_tests_averages = {}
        avg_pr_all_tests = (method_results["SKRIPT_TEST"]["precision"] + method_results["SKRIPT_FORMULA_TEST"]["precision"] + method_results["SLIDES_TEST"]["precision"]) / 3
        avg_rc_all_tests = (method_results["SKRIPT_TEST"]["recall"] + method_results["SKRIPT_FORMULA_TEST"]["recall"] +
                                method_results["SLIDES_TEST"]["recall"]) / 3
        avg_iou_all_tests = (method_results["SKRIPT_TEST"]["iou"] + method_results["SKRIPT_FORMULA_TEST"]["iou"] +
                                method_results["SLIDES_TEST"]["iou"]) / 3
        avg_f1_all_tests = (method_results["SKRIPT_TEST"]["f1"] + method_results["SKRIPT_FORMULA_TEST"]["f1"] +
                                method_results["SLIDES_TEST"]["f1"]) / 3
        all_tests_averages["precision"] = avg_pr_all_tests
        all_tests_averages["recall"] = avg_rc_all_tests
        all_tests_averages["iou"] = avg_iou_all_tests
        all_tests_averages["f1"] = avg_f1_all_tests

        # Coverage: wie viele Fragen lieferten überhaupt Chunks (nicht leer)?
        total_cases = sum(len(t) for t in tests.values())
        all_tests_averages["empty"] = method_empty
        all_tests_averages["total"] = total_cases
        all_tests_averages["coverage"] = (total_cases - method_empty) / total_cases if total_cases else 0.0

        method_results["overall"] = all_tests_averages
        results[method.value] = method_results
    return results, specific_results



def calculate_iou(ground_truth: list[str], retrieved: list[str]):
    gt = set(ground_truth)
    retrieved = list(retrieved)
    try:
        intersection = gt.intersection(retrieved)

        precision = len(intersection) / len(retrieved)
        recall = len(intersection) / len(gt)

        iou = (
            len(intersection)
            / (len(gt) + len(retrieved) - len(intersection))
        )
        f1 = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        if DEBUG:
            print('E: Division by zero')
        return 0,0,0,0
    return precision, recall, iou, f1

def string_to_tokens(some_string: str):
    tokens = some_string.lower().split()
    #print(tokens)
    return tokens

if __name__ == '__main__':

    n_values = [3, 5, 7]

    # Ergebnisse für alle n-Werte sammeln
    all_results = {}

    for n in n_values:
        results, specific_results = run_evaluation(
            dynamic=False,
            n_param=n
        )
        all_results[n] = results

    # Durchschnitt über n = 3, 5, 7 berechnen
    avg_results = {}

    # Methoden aus dem ersten Lauf übernehmen
    for method_name in all_results[n_values[0]]:
        metrics = {
            "precision": [],
            "recall": [],
            "iou": [],
            "f1": [],
            "coverage": []
        }

        for n in n_values:
            overall = all_results[n][method_name]["overall"]

            metrics["precision"].append(overall["precision"])
            metrics["recall"].append(overall["recall"])
            metrics["iou"].append(overall["iou"])
            metrics["f1"].append(overall["f1"])

            # Coverage als Anteil der nicht-leeren Ergebnisse
            coverage = (
                (overall["total"] - overall["empty"])
                / overall["total"]
                if overall["total"] > 0
                else 0
            )
            metrics["coverage"].append(coverage)

        # Mittelwerte berechnen
        avg_results[method_name] = {
            metric: sum(values) / len(values)
            for metric, values in metrics.items()
        }

    # Einzelne Ergebnisse ausgeben
    def _print_results(title, res):
        print(f'========== {title} ==========')

        for method_name, method in res.items():
            o = method["overall"]

            coverage = (
                (o["total"] - o["empty"]) / o["total"]
                if o["total"] > 0
                else 0
            )

            print(
                f"  {method_name:10s}  "
                f"P={o['precision']:.3f}  "
                f"R={o['recall']:.3f}  "
                f"IoU={o['iou']:.3f}  "
                f"F1={o['f1']:.3f}  "
                f"| coverage={o['total'] - o['empty']}/{o['total']} "
                f"(leer: {o['empty']})"
            )

    # Ergebnisse für jedes n
    for n in n_values:
        _print_results(f"STATIC (n={n})", all_results[n])

    # Durchschnitt aus n=3,5,7
    print()
    print("========== STATIC (AVERAGE n=3,5,7) ==========")

    for method_name, metrics in avg_results.items():
        print(
            f"  {method_name:10s}  "
            f"P={metrics['precision']:.3f}  "
            f"R={metrics['recall']:.3f}  "
            f"IoU={metrics['iou']:.3f}  "
            f"F1={metrics['f1']:.3f}  "
            f"| coverage={metrics['coverage']:.3f}"
        )