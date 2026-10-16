import unittest

from precision_budget.planner import Result, choose, sensitivity_order


class PlannerTests(unittest.TestCase):
    def test_rejects_fast_variant_that_exceeds_accuracy_budget(self):
        base = Result("fp32", (), .95, 10, 1000)
        fast_bad = Result("int8", (), .90, 2, 300)
        mixed = Result("mixed", ("a",), .945, 5, 600)
        self.assertEqual(choose(base, [fast_bad, mixed], .01), mixed)

    def test_falls_back_to_baseline_when_quantization_is_slower(self):
        base = Result("fp32", (), .95, 5, 1000)
        quant = Result("int8", (), .95, 8, 300)
        self.assertEqual(choose(base, [quant], .01), base)

    def test_near_tie_keeps_baseline(self):
        base = Result("fp32", (), .95, 10, 1000)
        quant = Result("int8", (), .95, 9.9, 300)
        self.assertEqual(choose(base, [quant], .01), base)

    def test_sensitivity_rank_uses_recovered_accuracy(self):
        full = Result("all", (), .8, 2, 100)
        probes = [Result("a", ("a",), .82, 3, 200), Result("b", ("b",), .9, 4, 200)]
        self.assertEqual(sensitivity_order(full, probes), ["b", "a"])


if __name__ == "__main__":
    unittest.main()
