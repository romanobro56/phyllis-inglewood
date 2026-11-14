from flies.chess.evaluate import compare, summarize
from flies.chess.selfplay import play_game
from flies.chess.tests.support import BrainTest


class EvaluationTests(BrainTest):
    def test_baselines_and_no_false_draw(self):
        white, black = self.brain(), self.brain()
        for lesion, random in [(False, True), (True, True), (False, False)]:
            report = compare(white, black, games=2, max_plies=4, candidate_lesion=lesion, random_opponent=random)
            self.assertEqual(report["games"], 2)
            self.assertEqual(report["illegal_attempts"], 0)
            self.assertEqual(report["draws"], 0)
            self.assertEqual(report["truncated"], 2)
            self.assertTrue(all(m["masked_legality_accuracy"] == 1 for m in report["per_stage"].values()))

    def test_true_draw_win_loss_counts(self):
        white, black = self.brain(), self.brain()
        checkmate, _ = play_game(white, black, fen="7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
        stalemate, _ = play_game(white, black, fen="7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")
        report = summarize([checkmate, checkmate, stalemate], ["white", "black", "white"])
        self.assertEqual([report[k] for k in ("wins", "losses", "draws", "truncated")], [1, 1, 1, 0])
