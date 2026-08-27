from scipy.stats import spearmanr

CORRECT_THRESHOLD = 4


def cohens_kappa(pairs: list[tuple[float, int]], threshold: int = CORRECT_THRESHOLD) -> float:
    n = len(pairs)
    if n == 0:
        raise ValueError("cohens_kappa requires at least one pair")

    a = b = c = d = 0
    for judge_score, human_score in pairs:
        judge_correct = judge_score >= threshold
        human_correct = human_score >= threshold
        if judge_correct and human_correct:
            a += 1
        elif judge_correct and not human_correct:
            b += 1
        elif not judge_correct and human_correct:
            c += 1
        else:
            d += 1

    po = (a + d) / n
    judge_correct_rate = (a + b) / n
    human_correct_rate = (a + c) / n
    pe = judge_correct_rate * human_correct_rate + (1 - judge_correct_rate) * (1 - human_correct_rate)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def calibration_report(pairs: list[tuple[float, int]]) -> dict:
    n = len(pairs)
    if n == 0:
        raise ValueError("calibration_report requires at least one (judge_score, human_score) pair")

    judge_scores = [j for j, _ in pairs]
    human_scores = [h for _, h in pairs]

    spearman_r, spearman_p = spearmanr(judge_scores, human_scores)
    mean_abs_diff = sum(abs(j - h) for j, h in pairs) / n

    return {
        "n": n,
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "cohens_kappa": cohens_kappa(pairs),
        "mean_abs_diff": mean_abs_diff,
    }
