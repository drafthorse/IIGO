"""CEC experiment settings and deterministic seed schedule."""

DIMENSION = 30
POPULATION_SIZE = 10
MAX_FES = 300_000

ALGORITHMS = (
    "IIGO", "PSO", "GWO", "SMA", "SCA", "RSA",
    "DE", "SNAKE", "RDWOA", "GCHHO", "LSHADE", "WEMFO", "CMA-ES",
    "CSO", "LSHADE_cnEpSin", "jSO",
)

BENCHMARK_IDS = {"CEC2014": 2014, "CEC2017": 2017, "CEC2022": 2022}

PAPER_FUNCTIONS = {
    "CEC2014": {2: 1, 3: 2, 7: 3, 8: 4, 10: 5, 11: 6,
                19: 7, 22: 8, 23: 9, 25: 10, 30: 11},
    "CEC2017": {5: 12, 7: 13, 8: 14, 9: 15, 11: 16, 16: 17,
                20: 18, 23: 19, 25: 20, 27: 21, 29: 22},
    "CEC2022": {1: 23, 2: 24, 3: 25, 7: 26, 8: 27, 10: 28,
                11: 29, 12: 30},
}

# Independent-run seed bases. Function and suite identifiers are
# mixed into these bases by derive_seed; every algorithm under the same
# (suite, function, run) condition receives the same final seed.
RUN_SEEDS = (
    104729, 130363, 155921, 181081, 205759, 230719,
    256019, 281117, 306457, 331777, 357031, 382343,
    407633, 432959, 458317, 483647, 508951, 534307,
    559691, 585043, 610447, 635807, 661163, 686521,
    711817, 737279, 762563, 787933, 813269, 838631,
)


def derive_seed(benchmark, function_number, run_number):
    if not 0 <= run_number < len(RUN_SEEDS):
        raise ValueError(f"run_number must be 0..{len(RUN_SEEDS)-1}")
    modulus = 2_147_483_647
    value = (RUN_SEEDS[run_number]
             + BENCHMARK_IDS[benchmark] * 1_000_003
             + function_number * 9_176)
    return int(value % modulus)
