"""Rules-only adapter. Black's canonical rank flip (square ^ 56) is self-inverse."""
import chess
import numpy as np

ACTIONS = [f"slot_{i}" for i in range(8)]
STAGES = ("source_file", "source_rank", "destination_file", "destination_rank", "promotion")
PROMOTIONS = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
OBS_DIM = 64
ENCODING = "chess-packed-v1-rank-flip"
SPATIAL_ENCODING = "chess-packed-v2-spatial"
STAGE_ONLY_ENCODING = "chess-stage-only-control-v1"
ENCODINGS = (ENCODING, SPATIAL_ENCODING, STAGE_ONLY_ENCODING)


def canonical_square(square, turn):
    return square if turn == chess.WHITE else square ^ 56


def decode_planes(obs):
    """Recover all 12 canonical piece planes exactly from the packed input."""
    packed = np.rint(np.asarray(obs[:32]) * 168).astype(int)
    codes = np.column_stack((packed % 13, packed // 13)).reshape(64)
    return np.stack([codes == i for i in range(1, 13)]).astype(np.float32)


class ChessEnv:
    def __init__(self, fen=chess.STARTING_FEN, max_plies=160, claim_draw=True):
        if max_plies < 1:
            raise ValueError("max_plies must be positive")
        self.board = chess.Board(fen)
        if not self.board.is_valid():
            raise ValueError("Invalid chess position")
        self.initial_fen = self.board.fen()
        self.max_plies = max_plies
        self.claim_draw = claim_draw
        self.plies = 0
        self.stage = 0
        self.choices = []
        self.truncated = False
        self.illegal_attempts = 0
        self._refresh()

    def _refresh(self):
        self._outcome = self.board.outcome(claim_draw=self.claim_draw)
        self._legal = tuple(self.board.legal_moves) if self._outcome is None else ()

    @property
    def done(self):
        return self._outcome is not None or self.truncated

    def outcome(self):
        return dict(result=self._outcome.result() if self._outcome else "*",
                    termination=self._outcome.termination.name if self._outcome else None,
                    winner=("white" if self._outcome.winner else "black")
                    if self._outcome and self._outcome.winner is not None else None,
                    truncated=self.truncated, claim_draw=self.claim_draw)

    def coordinates(self, move):
        src = canonical_square(move.from_square, self.board.turn)
        dst = canonical_square(move.to_square, self.board.turn)
        coords = [chess.square_file(src), chess.square_rank(src),
                  chess.square_file(dst), chess.square_rank(dst)]
        if move.promotion:
            coords.append(PROMOTIONS.index(move.promotion))
        return coords

    def candidates(self):
        return [m for m in self._legal if self.coordinates(m)[:self.stage] == self.choices]

    def mask(self):
        mask = np.zeros(8, dtype=bool)
        if not self.done:
            for move in self.candidates():
                mask[self.coordinates(move)[self.stage]] = True
            if not mask.any():
                raise RuntimeError("Nonterminal hierarchy has no legal continuation")
        return mask

    def observe(self, encoding=ENCODING):
        if encoding not in ENCODINGS:
            raise ValueError("Unknown chess observation encoding")
        codes = np.zeros(64, dtype=np.int32)
        for sq, p in self.board.piece_map().items():
            codes[canonical_square(sq, self.board.turn)] = p.piece_type + (0 if p.color == self.board.turn else 6)
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[:32] = (codes[::2] + 13 * codes[1::2]) / 168.0
        obs[32 + self.stage] = 1
        obs[37:37 + min(4, len(self.choices))] = [(v + 1) / 8 for v in self.choices[:4]]
        us, them = self.board.turn, not self.board.turn
        obs[41:45] = [self.board.has_kingside_castling_rights(us), self.board.has_queenside_castling_rights(us),
                      self.board.has_kingside_castling_rights(them), self.board.has_queenside_castling_rights(them)]
        obs[45] = (canonical_square(self.board.ep_square, us) + 1) / 64 if self.board.ep_square is not None else 0
        obs[46] = self.board.is_check()
        obs[47] = min(self.board.halfmove_clock, 150) / 150
        obs[48] = us == chess.WHITE
        obs[49] = max(0, 1 - self.plies / self.max_plies)
        obs[50] = self.board.is_repetition(2)
        obs[51] = self.board.is_repetition(3)
        obs[52] = self.claim_draw
        if encoding == STAGE_ONLY_ENCODING:
            # Explicit scientific ablation, never the normal task observation.
            obs[:32] = 0
            obs[37:] = 0
        if encoding == SPATIAL_ENCODING:
            # Redundant board geometry, no tactical oracle or legal-move ranking.
            # All 32 packed board floats remain intact, also in full chess games.
            squares = [self.board.king(us), self.board.king(them)]
            for piece_type in (chess.QUEEN, chess.ROOK):
                pieces = sorted(canonical_square(s, us) for s in self.board.pieces(piece_type, us))
                squares.append(canonical_square(pieces[0], us) if pieces else None)
            for i, square in enumerate(squares):
                if square is not None:
                    square = canonical_square(square, us)
                    obs[53 + 2*i:55 + 2*i] = [(chess.square_file(square)+1)/8,
                                              (chess.square_rank(square)+1)/8]
            if len(self.choices) >= 2:
                src = chess.square(self.choices[0], self.choices[1])
                p = self.board.piece_at(canonical_square(src, us))
                obs[61] = p.piece_type / 6 if p else 0
                king = canonical_square(self.board.king(them), us)
                obs[62:64] = [(chess.square_file(king)-chess.square_file(src)+7)/14,
                              (chess.square_rank(king)-chess.square_rank(src)+7)/14]
        return obs

    def select(self, action):
        if self.done:
            raise RuntimeError("No neural decision is permitted after termination")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 8 or not self.mask()[action]:
            self.illegal_attempts += 1
            raise ValueError(f"Illegal stage choice {action} at {STAGES[self.stage]}")
        self.choices.append(int(action))
        self.stage += 1
        moves = self.candidates()
        if self.stage == 4 and any(m.promotion for m in moves):
            return None
        if self.stage < 4:
            return None
        if len(moves) != 1:
            raise RuntimeError("Completed hierarchy is ambiguous")
        return self.push(moves[0])

    def push(self, move):
        """Apply a fully specified move, including the explicitly random demo reply."""
        if self.done or move not in self.board.legal_moves:
            self.illegal_attempts += 1
            raise ValueError(f"Rejected full move: {move}")
        event = dict(uci=move.uci(), san=self.board.san(move), before=self.board.fen(),
                     side="white" if self.board.turn else "black", legal=True)
        self.board.push(move)
        self.plies += 1
        self.stage, self.choices = 0, []
        self._refresh()
        self.truncated = self._outcome is None and self.plies >= self.max_plies
        event.update(fen=self.board.fen(), ply=self.plies, outcome=self.outcome())
        return event
