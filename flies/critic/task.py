"""Critic fly (Gordo Inglewood): a fruit fly food critic. Serve it a dish, it tastes it and gives a verdict.

Every serving is described by what a fly can actually sense (sugar, fruit esters, fermentation,
rot, poop smell, bitterness, ...) plus the food group. The fly answers with one of 8 verdicts,
each read from its own fixed group of real descending neurons. The precise 0-10 score shown in
the tasting room (flies/critic/tasting.py) is the expected verdict value under the brain's own
verdict probabilities: sum_i p_i * VALUES[i], so the decimals come straight from neuron firing.

Training is teacher mode with a *soft* teacher: for a serving whose palate score is 8.9 the
teacher says "delicious" (8.57) or "chefs_kiss" (10) with the mix that averages to 8.9. Copying
that teaches calibrated probabilities, so the expected value lands on the palate score.
"""
import numpy as np

from core.task import FlyTask

# --- the menu ---------------------------------------------------------------------------------
GROUPS = ["fruit", "vegetable", "meat", "rotten", "poop"]
FEATURES = ["sugar", "fruit_esters", "fermentation", "rot", "mold", "poop_smell", "protein", "fat",
            "bitterness", "fiber", "salt", "acidity", "moisture", "spice"]

# name: (group, emoji, palate score 0-10, feature profile {feature: 0..1})
MENU = {
    # fruit: a fruit fly's whole personality. Always scores high (>= 8).
    "banana":       ("fruit", "🍌", 9.90, dict(sugar=.85, fruit_esters=.95, fermentation=.25, moisture=.6, fiber=.2)),
    "mango":        ("fruit", "🥭", 9.60, dict(sugar=.9, fruit_esters=.9, fermentation=.15, moisture=.75, acidity=.2)),
    "grapes":       ("fruit", "🍇", 9.50, dict(sugar=.85, fruit_esters=.8, fermentation=.2, moisture=.8, acidity=.3)),
    "peach":        ("fruit", "🍑", 9.40, dict(sugar=.8, fruit_esters=.85, moisture=.8, acidity=.2, fermentation=.1)),
    "strawberry":   ("fruit", "🍓", 9.30, dict(sugar=.7, fruit_esters=.9, moisture=.8, acidity=.35)),
    "cherry":       ("fruit", "🍒", 9.25, dict(sugar=.75, fruit_esters=.8, moisture=.75, acidity=.35)),
    "blueberries":  ("fruit", "🫐", 9.15, dict(sugar=.65, fruit_esters=.7, moisture=.7, acidity=.35, fiber=.2)),
    "pineapple":    ("fruit", "🍍", 9.05, dict(sugar=.75, fruit_esters=.85, moisture=.8, acidity=.6, fiber=.3)),
    "watermelon":   ("fruit", "🍉", 8.95, dict(sugar=.6, fruit_esters=.6, moisture=1., fiber=.1)),
    "apple":        ("fruit", "🍎", 8.85, dict(sugar=.65, fruit_esters=.7, moisture=.7, acidity=.3, fiber=.35)),
    "orange":       ("fruit", "🍊", 8.70, dict(sugar=.6, fruit_esters=.75, moisture=.85, acidity=.6, bitterness=.1)),
    "lemon":        ("fruit", "🍋", 8.20, dict(sugar=.2, fruit_esters=.8, moisture=.8, acidity=1., bitterness=.2)),
    # vegetables: bitter, fibrous, not sweet. Mostly a polite "meh" or worse.
    "corn":         ("vegetable", "🌽", 5.40, dict(sugar=.5, moisture=.6, fiber=.5, protein=.15)),
    "tomato":       ("vegetable", "🍅", 6.90, dict(sugar=.35, fruit_esters=.45, moisture=.9, acidity=.55)),
    "carrot":       ("vegetable", "🥕", 4.30, dict(sugar=.4, moisture=.6, fiber=.6, bitterness=.1)),
    "cucumber":     ("vegetable", "🥒", 4.00, dict(sugar=.1, moisture=1., fiber=.3, bitterness=.15)),
    "bell pepper":  ("vegetable", "🫑", 3.60, dict(sugar=.25, moisture=.8, fiber=.4, bitterness=.25)),
    "lettuce":      ("vegetable", "🥬", 3.10, dict(sugar=.05, moisture=.9, fiber=.7, bitterness=.3)),
    "potato":       ("vegetable", "🥔", 2.90, dict(sugar=.1, moisture=.5, fiber=.4, protein=.1)),
    "broccoli":     ("vegetable", "🥦", 2.20, dict(sugar=.05, moisture=.6, fiber=.8, bitterness=.75)),
    "onion":        ("vegetable", "🧅", 1.50, dict(sugar=.2, moisture=.7, fiber=.3, bitterness=.5, spice=.5)),
    "jalapeño":     ("vegetable", "🌶️", 1.10, dict(sugar=.1, moisture=.6, fiber=.4, bitterness=.3, spice=1.)),
    "garlic":       ("vegetable", "🧄", 0.60, dict(sugar=.05, moisture=.4, fiber=.3, bitterness=.6, spice=.85)),
    # meat: a decent amount of protein, but a fruit fly isn't a carnivore.
    "burger":       ("meat", "🍔", 5.10, dict(protein=.6, fat=.6, salt=.6, sugar=.2, moisture=.5)),
    "bacon":        ("meat", "🥓", 4.90, dict(protein=.6, fat=.9, salt=.9, moisture=.2)),
    "sausage":      ("meat", "🌭", 4.40, dict(protein=.6, fat=.75, salt=.75, moisture=.45, spice=.2)),
    "steak":        ("meat", "🥩", 4.20, dict(protein=.95, fat=.45, salt=.2, moisture=.6)),
    "chicken leg":  ("meat", "🍗", 3.80, dict(protein=.9, fat=.4, salt=.4, moisture=.5)),
    "sashimi":      ("meat", "🍣", 3.40, dict(protein=.85, fat=.35, salt=.3, moisture=.8)),
    # rotten: fermenting fruit is a fly delicacy; spoiled everything else, much less so.
    "overripe banana": ("rotten", "🍌", 8.10, dict(sugar=.9, fruit_esters=.8, fermentation=.95, rot=.3, moisture=.7)),
    "fermented apple": ("rotten", "🍎", 7.60, dict(sugar=.6, fruit_esters=.6, fermentation=.9, rot=.35, acidity=.5, moisture=.6)),
    "compost heap":    ("rotten", "🥀", 5.60, dict(sugar=.3, fruit_esters=.3, fermentation=.6, rot=.6, mold=.4, fiber=.5, moisture=.7)),
    "rotten meat":     ("rotten", "🍖", 4.70, dict(protein=.8, fat=.4, rot=1., poop_smell=.3, moisture=.6)),
    "rotten egg":      ("rotten", "🥚", 3.50, dict(protein=.6, rot=.85, poop_smell=.4, moisture=.5)),
    "spoiled milk":    ("rotten", "🥛", 2.80, dict(protein=.3, fat=.3, fermentation=.4, rot=.6, acidity=.8, moisture=1.)),
    "moldy bread":     ("rotten", "🍞", 2.10, dict(sugar=.2, mold=1., rot=.3, fiber=.4, moisture=.3)),
    # poop: a fly's guilty pleasure. It won't call it fine dining, but it doesn't hate it either.
    "cow pie":         ("poop", "💩", 6.30, dict(poop_smell=.85, fermentation=.4, fiber=.7, rot=.4, moisture=.7)),
    "dog poop":        ("poop", "💩", 5.70, dict(poop_smell=1., protein=.3, rot=.5, moisture=.5)),
    "horse apples":    ("poop", "💩", 5.90, dict(poop_smell=.8, fiber=.85, fermentation=.35, rot=.35, moisture=.5)),
    "bird droppings":  ("poop", "💩", 4.60, dict(poop_smell=.7, acidity=.4, rot=.3, salt=.3, moisture=.6)),
}
FOODS = list(MENU)

# small per-serving palate nudges: how much each sensed feature (relative to the dish's usual
# profile) moves the score. A riper, sweeter banana scores a hair higher; a moldier one lower.
PALATE = dict(sugar=1.2, fruit_esters=1.0, fermentation=0.6, rot=-0.8, mold=-1.4, poop_smell=0.2,
              protein=0.2, fat=0.0, bitterness=-1.3, fiber=-0.4, salt=-0.3, acidity=-0.3,
              moisture=0.3, spice=-1.2)
PALATE_W = np.array([PALATE[f] for f in FEATURES], np.float32)

VALUES = np.linspace(0.0, 10.0, 8)   # score attached to each verdict: 0, 1.43, ..., 10
SERVINGS = 20                        # dishes per episode


def profile(food):
    return np.array([MENU[food][3].get(f, 0.0) for f in FEATURES], np.float32)


def serve(food, rng):
    """One serving of a dish: its usual profile plus natural variation (ripeness, freshness)."""
    base = profile(food)
    feats = np.clip(base + rng.normal(0.0, 0.06, len(FEATURES)).astype(np.float32), 0.0, 1.0)
    return feats


def palate_score(food, feats):
    """The ground-truth critic: dish score nudged by how this particular serving tastes."""
    group, _, base, _ = MENU[food]
    s = base + float(PALATE_W @ (feats - profile(food)))
    lo = 8.0 if group == "fruit" else 0.0            # fruit always scores high
    return float(np.clip(s, lo, 10.0))


def rating(probs):
    """Precise 0-10 rating from the brain's verdict probabilities (expected verdict value)."""
    p = np.asarray(probs, np.float64)
    return float((p / p.sum()) @ VALUES)


# Fixed score bands, so the word always matches the number: each verdict owns the scores
# closest to its value. 0-0.71 spit it out, 0.71-2.14 revolting, 2.14-3.57 bad, 3.57-5.00 meh,
# 5.00-6.43 decent, 6.43-7.86 good, 7.86-9.29 delicious, 9.29-10 chef's kiss.
BAND_EDGES = (VALUES[:-1] + VALUES[1:]) / 2


def verdict_for(score):
    """Index of the verdict whose band contains this score."""
    return int(np.searchsorted(BAND_EDGES, score, side="right"))


class CriticFly(FlyTask):
    name = "critic"
    actions = ["spit_it_out", "revolting", "bad", "meh", "decent", "good", "delicious", "chefs_kiss"]
    obs_dim = len(FEATURES) + len(GROUPS)            # 19
    mode = "teacher"
    max_steps = SERVINGS
    greedy = True                    # in the world, say the brain's top verdict (no dice roll)
    train = {"teacher_episodes": 600, "teacher_epsilon": 0.0, "epochs": 40, "lr": 1e-3}
    # in the world the critic doesn't eat at random: it waits for someone to pick a dish off this menu
    menu = [dict(name=n, emoji=e, group=g) for n, (g, e, _, _) in MENU.items()]

    def choose(self, item):
        if item not in MENU:
            raise KeyError(item)
        self.new_dish(item)
        return self.obs()

    def reset(self, rng):
        self.rng = rng
        self.n = 0
        self.last = None
        self.new_dish()
        return self.obs()

    def new_dish(self, food=None):
        self.food = food if food is not None else FOODS[int(self.rng.integers(len(FOODS)))]
        self.feats = serve(self.food, self.rng)
        self.score = palate_score(self.food, self.feats)

    def obs(self):
        g = np.zeros(len(GROUPS), np.float32)
        g[GROUPS.index(MENU[self.food][0])] = 1.0
        return np.concatenate([self.feats, g]).astype(np.float32)

    def teacher(self):
        # soft label: split between the two verdicts that bracket the palate score
        x = self.score / 10.0 * (len(VALUES) - 1)
        lo = int(np.floor(x))
        if lo >= len(VALUES) - 1:
            return len(VALUES) - 1
        return lo + int(self.rng.random() < (x - lo))

    def step(self, action, rng):
        err = abs(VALUES[action] - self.score)
        reward = max(-1.0, 1.0 - err / (VALUES[1] - VALUES[0]))   # 1 = spot on, 0 = one verdict off
        self.last = (self.food, action, self.score)
        self.n += 1
        self.new_dish()
        return self.obs(), reward, self.n >= self.max_steps, {}

    def caption(self, action):
        food, a, _ = self.last
        emoji = MENU[food][1]
        lines = ["SPITS IT OUT", "calls it revolting", "says bad", "shrugs: meh",
                 "says decent", "says good", "says delicious", "CHEF'S KISS"]
        return f"{emoji} {food}: {lines[a]} ({VALUES[a]:.1f}/10)"
