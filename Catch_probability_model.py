#Run this: pip3 install joblib pandas numpy scikit-learn xgboost matplotlib

import sys
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Constants 

DEFAULT_CSV = "data/statcast_2021_outfield.csv"

# Path where the trained XGBoost model is saved after training 
# I was going to add joblib reload function to streamline singular predictions
# I ran out of time, unfortunately
MODEL_SAVE_PATH = "catch_prob_xgb.pkl"

# Statcast coordinate origin — approximately home plate
# (hc_x=125, hc_y=199 in Baseball Savant's pixel coordinate system)
HOME_PLATE_X = 125
HOME_PLATE_Y = 199

# Distance threshold (in Statcast units) beyond which we flag "near wall"
# Typical outfield fence sits around 280–350 ft; 270 units is a safe proxy
WALL_PROXIMITY_THRESHOLD = 270

# The 8 features fed to both models — order matters for joblib reload
FEATURES = [
    "distance_needed",      # how far the fielder had to travel (ft)
    "opportunity_time",     # seconds from pitch release to landing
    "direction",            # spray angle in degrees (0 = straightaway CF)
    "launch_angle",         # what angle does the initial hit create compared to the ground 
    "near_wall",            # 1 if ball landed near the warning track
    "running_back",         # 1 if fielder ran primarily backward
    "running_lateral",      # 1 if fielder ran primarily sidetoside
    "is_line_drive",        # 1 if bb_type == line_drive (harder to read)
    "time_distance_ratio",  # opportunity_time / distance — low = hardest
]

# Statcast star rating thresholds (catch probability → difficulty stars)
# 5 stars = hardest play; routine plays get no star at all
STAR_THRESHOLDS = [
    (0.25, 5),
    (0.50, 4),
    (0.75, 3),
    (0.90, 2),
    (0.95, 1),
]


#  1. Data Loading & Cleaning 

def load_data(filepath: str) -> pd.DataFrame:
    """
    Read the raw Statcast CSV exported from Baseball Savant.

    """
    df = pd.read_csv(filepath, low_memory=False)
    print(f"[load]  {len(df):>6,} rows loaded from '{filepath}'")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    #may be redundant now, as filters should've stopped most of incorrect entries
    """
    Filter to outfield fly-ball / line-drive opportunities only, and
    drop rows missing the coordinates we need for feature engineering.

    Statcast hit_location codes: 7 = LF, 8 = CF, 9 = RF.
    """
    # Keep only fly balls and line drives — ground balls can't be caught
    df = df[df["bb_type"].isin(["fly_ball", "line_drive"])].copy()

    # Keep only plays where the primary fielder was an outfielder
    df = df[df["hit_location"].isin([7, 8, 9])].copy()

    # Drop rows where we can't compute features (foul balls, etc.)
    required_cols = ["hc_x", "hc_y", "hit_distance_sc", "launch_angle",
                     "launch_speed"]
    df = df.dropna(subset=required_cols)

    return df


#  2. Feature Engineering 

def estimate_hang_time(
    launch_angle: pd.Series,
    launch_speed: pd.Series,
) -> pd.Series:
    """
    Physics-based hang time estimate, as Statcast's measured value was absent in 2021.

    Uses simple projectile motion: t = 2 * v_y / g
      where v_y = v * sin(θ) is the vertical component of exit velocity.

    """
    g_ft_s2 = 32.174                            # gravitational acceleration
    v_fps   = launch_speed * 1.467              # mph → ft/s conversion
    theta   = np.radians(launch_angle.clip(-10, 60))
    v_vert  = v_fps * np.sin(theta)
    return (2 * v_vert / g_ft_s2).clip(0, 8)

def estimate_opportunity_time(launch_angle, bbdist):
    return max(1.5, min(7.0, 2 + (bbdist / 120) + (launch_angle / 20)))

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the target column and all 8 model features from raw Statcast fields.

    The 4 features Statcast officially uses:
        1. Distance needed  (how far the outfielder ran)
        2. Opportunity time (how long the ball was in the air)
        3. Direction        (back / lateral / in)
        4. Wall proximity   (was the fence a factor?)

    We add 4 extras:
        5. is_line_drive       (harder to read off the bat)
        6. running_back flag   (derived from direction)
        7. running_lateral flag(derived from direction)
        8. time_distance_ratio (low ratio = hardest plays)
    """
    # ── Target label ─────────────────────────────────────────────────────────
    # "caught" means the fielder recorded an out on this specific play
    caught_events = ["field_out", "sac_fly", "sac_fly_double_play"]
    df["caught"] = df["events"].isin(caught_events).astype(int)
    print(f"[feat]  overall catch rate: {df['caught'].mean():.1%}")

    # ── Feature 1: Distance Needed ───────────────────────────────────────────
    # hit_distance_sc = total ball-travel distance; good proxy for fielder
    # distance on plays that stay in the park
    df["distance_needed"] = (
        df["hit_distance_sc"].fillna(df["hit_distance_sc"].median())
    )

    # Opportunity Time 
    # Prefer Statcast's measured hang_time; fall back to physics estimate
    has_hang = (
        "hang_time" in df.columns
        and df["hang_time"].notna().sum() > len(df) * 0.5
    )
    if has_hang:
        df["opportunity_time"] = df["hang_time"].fillna(
            estimate_hang_time(df["launch_angle"], df["launch_speed"])
        )
    else:
        # No hang_time column → estimate entirely from launch conditions
        df["opportunity_time"] = estimate_hang_time(
            df["launch_angle"], df["launch_speed"]
        )

    # Direction 
    # Spray angle: 0° = straight CF, negative = pull, positive = opposite
    if "spray_angle" in df.columns:
        df["direction"] = df["spray_angle"].fillna(0)
    else:
        # Derive from landing coordinates relative to home plate
        df["direction"] = np.degrees(
            np.arctan2(df["hc_x"] - HOME_PLATE_X, HOME_PLATE_Y - df["hc_y"])
        )

    # Flag the two hardest directional categories
    abs_dir = df["direction"].abs()
    df["running_back"]    = (abs_dir < 20).astype(int)
    df["running_lateral"] = ((abs_dir >= 20) & (abs_dir < 60)).astype(int)

    # Feature 4: Wall Proximity 
    # Euclidean distance from home plate to landing spot (Statcast units)
    df["dist_from_home"] = np.sqrt(
        (df["hc_x"] - HOME_PLATE_X) ** 2
        + (df["hc_y"] - HOME_PLATE_Y) ** 2
    )
    df["near_wall"] = (df["dist_from_home"] > WALL_PROXIMITY_THRESHOLD).astype(int)

    # Extra features 
    # Line drives arrive faster and are harder to read off the bat
    df["is_line_drive"] = (df["bb_type"] == "line_drive").astype(int)

    # Low ratio = little time to cover a lot of ground = hardest plays
    df["time_distance_ratio"] = (
        df["opportunity_time"] / (df["distance_needed"] + 1)
    )

    return df


# 3. Train / Test Split 

def split_data(df: pd.DataFrame):
    """
    80 / 20 stratified split so the catch rate is the same in both halves.
    Stratifying matters here because ~78% of fly balls are caught —
    a naive split could skew the test set.

    """
    X = df[FEATURES]
    y = df["caught"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(
        f"[split] train={len(X_train):,}  test={len(X_test):,}  "
        f"| catch rate  train={y_train.mean():.1%}  test={y_test.mean():.1%}"
    )
    return X_train, X_test, y_train, y_test


# 4. Model Training 

def train_logistic(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Logistic Regression baseline.

    We scale features first (required for LR) and use class_weight='balanced'
    to compensate for the ~78/22 caught/not-caught imbalance.

    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # penalizes missed "not-caught" predictions
        random_state=42,
    )
    lr.fit(X_scaled, y_train)

    # Print signed coefficients — positive = raises catch probability
    coef_df = pd.DataFrame(
        {"feature": FEATURES, "coefficient": lr.coef_[0]}
    ).sort_values("coefficient", ascending=False)
    print("\n[lr]    coefficients (positive → increases catch prob):")
    print(coef_df.to_string(index=False))

    return lr, scaler


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    """
    XGBoost gradient-boosted classifier — the main model.

    Key choices:
      - max_depth=4    : shallow trees reduce overfitting on ~20k rows
      - learning_rate=0.05: slow learner + more trees → better calibration
      - scale_pos_weight: corrects for class imbalance without oversampling
      - early_stopping : uses a validation slice to stop before overfitting

    """
    # Ratio of negative to positive class → tells XGBoost how much to
    # up-weight the minority class (not-caught plays)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,           # row-level bagging per tree
        colsample_bytree=0.8,    # feature-level bagging per tree
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        early_stopping_rounds=20,
    )

    # Hold out 15% of training data as a validation set for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    print(f"[xgb]   best iteration: {model.best_iteration}")

    return model


#  5. Evaluation 

def evaluate_model(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scaler: StandardScaler = None,
) -> np.ndarray:
    """
    Print AUC-ROC, log loss, Brier score, and a classification report.

    AUC-ROC  → overall discrimination (can we rank plays by difficulty?)
    Log Loss → penalizes confident wrong predictions heavily
    Brier    → mean squared error of probabilities (0 = perfect)

    """
    X_eval = scaler.transform(X_test) if scaler else X_test
    probs  = model.predict_proba(X_eval)[:, 1]
    preds  = (probs >= 0.5).astype(int)

    auc   = roc_auc_score(y_test, probs)
    ll    = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  AUC-ROC : {auc:.4f}   (0.5 = coin flip, 1.0 = perfect)")
    print(f"  Log Loss: {ll:.4f}   (lower = better)")
    print(f"  Brier   : {brier:.4f}   (0 = perfect probability estimates)")
    print()
    print(classification_report(
        y_test, preds, target_names=["Not Caught", "Caught"]
    ))
    return probs


def plot_evaluation(
    models_probs: dict,
    y_test: pd.Series,
    xgb_model,
    X_test: pd.DataFrame,
) -> None:
    """
    Three-panel evaluation figure:
      Left  : ROC curves for both models
      Center: Calibration curves — are the probabilities trustworthy?
      Right : XGBoost feature importance

    The calibration plot is the most important for this project because
    we're outputting probabilities, not just classes. A predicted 40%
    should happen roughly 40% of the time in real life.

    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Catch Probability — Model Evaluation", fontsize=14)

    # ROC Curves 
    ax = axes[0]
    for label, probs in models_probs.items():
        RocCurveDisplay.from_predictions(y_test, probs, name=label, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.5)")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=9)

    # Calibration Curves
    ax = axes[1]
    for label, probs in models_probs.items():
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", label=label, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction actually caught")
    ax.set_title("Calibration Curve\n(diagonal = perfect calibration)")
    ax.legend(fontsize=9)

    #Feature Importance
    ax = axes[2]
    feat_imp = (
        pd.Series(xgb_model.feature_importances_, index=FEATURES)
        .sort_values()
    )
    feat_imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Importance score")

    plt.tight_layout()
    out_path = "catch_probability_evaluation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[plot]  saved evaluation figure → {out_path}")
    plt.show()


# ── 6. Single-Play Prediction Helpers ────────────────────────────────────────

def get_star_rating(catch_prob: float) -> str:
    """
    Convert a catch probability to Statcast's difficulty star rating.

    Statcast's system:
      ≤25%  → 5 stars (hardest)
      ≤50%  → 4 stars
      ≤75%  → 3 stars
      ≤90%  → 2 stars
      ≤95%  → 1 star
      >95%  → routine (no star awarded)

    Args:
        catch_prob: Model output in [0, 1].

    Returns:
        Human-readable star rating string.
    """
    for threshold, stars in STAR_THRESHOLDS:
        if catch_prob <= threshold:
            return "★" * stars + f"  ({stars}-star difficulty)"
    return "routine  (no star)"


def build_play_from_inputs(
    launch_angle: float,
    bbdist: float,
    direction: float,
    near_wall: int,
    is_line_drive: int,
    fielder_hit_to: int,
) -> dict:
    """
    Construct all trained model features from publicly accessible
    Baseball Savant inputs.

    """

    abs_dir = abs(direction)

    # Approximate distance the fielder ran
    # CF generally covers more ground
    if fielder_hit_to == 8:  # Center field
        distance_needed = bbdist * 0.35
    else:  # LF / RF
        distance_needed = bbdist * 0.28

    # Approximate hang/opportunity time
    opportunity_time = max(
        1.5,
        min(
            7.0,
            2 + (bbdist / 120) + (launch_angle / 20)
        )
    )

    return {
        "distance_needed": distance_needed,
        "opportunity_time": opportunity_time,
        "direction": direction,
        "launch_angle": launch_angle,
        "near_wall": near_wall,
        "running_back": int(abs_dir < 20),
        "running_lateral": int(20 <= abs_dir < 60),
        "is_line_drive": is_line_drive,
        "time_distance_ratio": (
            opportunity_time / (distance_needed + 1)
        ),
    }

def predict_single_play(model, play: dict) -> float:
    """
    Run the model on one play dictionary and return the catch probability.

    """
    X = pd.DataFrame([play])[FEATURES]
    return float(model.predict_proba(X)[0][1])


def display_prediction(prob: float) -> None:
    """
    Print a formatted prediction result to the terminal.

    """
    stars = get_star_rating(prob)
    bar   = build_prob_bar(prob)
    print()
    print(f"  Catch Probability : {prob:.1%}")
    print(f"  Difficulty        : {stars}")
    print(f"  {bar}")
    print()


def build_prob_bar(prob: float, width: int = 30) -> str:
    """
    Render a simple progress bar showing catch probability.

    Example: [████████████░░░░░░░░░░░░░░░░░░]  40%

    """
    filled = int(round(prob * width))
    empty  = width - filled
    return f"[{'█' * filled}{'░' * empty}]  {prob:.0%}"


# ── 7. Interactive CLI ────────────────────────────────────────────────────────

def prompt_yes_no(question: str) -> bool:
    """
    Ask a yes/no question and return True for 'y', False for 'n'.
    Loops until the user gives a valid answer.

    """
    while True:
        answer = input(f"{question} [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please enter y or n.")


def prompt_float(label: str, lo: float, hi: float) -> float:
    """
    Prompt for a numeric value within [lo, hi], re-asking on bad input.
    """
    while True:
        raw = input(f"  {label} ({lo}–{hi}): ").strip()
        try:
            val = float(raw)
            if lo <= val <= hi:
                return val
            print(f"  Please enter a value between {lo} and {hi}.")
        except ValueError:
            print("  That doesn't look like a number. Try again.")


def run_interactive_cli(model) -> None:
    """
    Interactive loop: user enters publicly accessible Baseball Savant
    metrics for a play and receives a predicted catch probability.

    Inputs are designed to match information commonly available
    from Baseball Savant or Statcast pages. (some inputs are proxy, 
    but are mostly accurate)

    """
    print()
    print("=" * 58)
    print("  MLB Catch Probability Predictor")
    print("  Simplified Baseball Savant Statcast Model")
    print("=" * 58)
    print("  Enter Baseball Savant play details.")
    print("  Type Ctrl-C or answer 'n' to exit.")
    print()

    while True:
        print("─" * 58)

        #  Collect inputs 
        launch_angle = prompt_float(
            "Launch angle (degrees)", -10, 60
        )

        bbdist = prompt_float(
            "Estimated ball distance (ft)", 50, 500
        )

        direction = prompt_float(
            "Attack direction (deg, −=pull, +=oppo)", -90, 90
        )

        fielder_hit_to = int(prompt_float(
            "Fielder hit to (LF=7, CF=8, RF=9)", 7, 9
        ))

        near_wall = int(
            prompt_yes_no("Near the warning track?")
        )

        is_line_drive = int(
            prompt_yes_no("Was it a line drive?")
        )

        # ── Build derived features & predict ─────────────────────
        play = build_play_from_inputs(
            launch_angle=launch_angle,
            bbdist=bbdist,
            direction=direction,
            near_wall=near_wall,
            is_line_drive=is_line_drive,
            fielder_hit_to=fielder_hit_to,
        )

        prob = predict_single_play(model, play)
        display_prediction(prob)

        #  Continue? 
        if not prompt_yes_no("Enter another play?"):
            print(
                "\n  Thanks for using Grayson's Catch "
                "Probability Predictor!\n"
            )
            break


#  8. Main 

def main() -> None:


    #  Determine CSV path 
    csv_path = DEFAULT_CSV
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            csv_path = arg   # allow: python script.py my_data.csv

    #  Full training pipeline 
    df = load_data(csv_path)
    df = clean_data(df)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    print("\n[train] Logistic Regression (baseline) ...")
    lr_model, scaler = train_logistic(X_train, y_train)

    print("\n[train] XGBoost ...")
    xgb_model = train_xgboost(X_train, y_train)

    #  Evaluate both models 
    lr_probs  = evaluate_model(
        "Logistic Regression", lr_model, X_test, y_test, scaler=scaler
    )
    xgb_probs = evaluate_model(
        "XGBoost", xgb_model, X_test, y_test
    )

    plot_evaluation(
        {"Logistic Regression": lr_probs, "XGBoost": xgb_probs},
        y_test,
        xgb_model,
        X_test,
    )

    #  Save model
    joblib.dump(xgb_model, MODEL_SAVE_PATH)
    print(f"[save]  model saved → {MODEL_SAVE_PATH}")

    model = xgb_model   # hand off to the CLI below


    run_interactive_cli(model)


if __name__ == "__main__":
    main()
