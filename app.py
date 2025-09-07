from flask import Flask, render_template, request, redirect, url_for, session, g, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField, SelectField, DateField, BooleanField
from wtforms.validators import DataRequired, Email, Length, EqualTo, Optional
from werkzeug.security import generate_password_hash, check_password_hash
from flask_babel import Babel, gettext as _
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pickle, json, os, uuid, shap, csv
import uuid as _uuid
from functools import wraps
from io import StringIO
from wtforms.validators import Optional as Opt
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-me-in-prod")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.config["BABEL_DEFAULT_LOCALE"] = "en"
app.config["BABEL_DEFAULT_TIMEZONE"] = "Asia/Colombo"
babel = Babel()
babel.init_app(app, locale_selector=lambda: session.get("lang", "en"))

LANGUAGES = {"en": "English", "si": "සිංහල"}

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

INPUT_SCALE_STARTS_AT_ONE = True
CONSENT_VERSION = "v1"
CONSENT_RETENTION = "12 months"
CONSENT_CONTACT_EMAIL = "kalana@example.com"
MODEL_VERSION = "LR_v1"

ADMIN_EMAILS = {
    e.strip().lower()
    for e in os.getenv("ADMIN_EMAILS", "").split(",")
    if e.strip()
}

STATIC_PLOTS_DIR = Path(app.root_path) / "static" / "plots"
STATIC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def is_admin() -> bool:
    return (
        current_user.is_authenticated
        and getattr(current_user, "email", None)
        and current_user.email.lower() in ADMIN_EMAILS
    )

def current_session_id():
    sid = session.get("anon_id")
    if not sid:
        sid = str(uuid.uuid4())
        session["anon_id"] = sid
    return sid

class User(db.Model, UserMixin):
    __tablename__ = "users"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(_uuid.uuid4()))
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    name = db.Column(db.String(120), nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    def set_password(self, raw_password: str):
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)

class Assessment(db.Model):
    __tablename__ = "assessments"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(_uuid.uuid4()))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    session_id = db.Column(db.String(64), index=True, nullable=True)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=True)
    consent_version = db.Column(db.String(16), nullable=True)
    model_version = db.Column(db.String(32), nullable=False, default=MODEL_VERSION)
    dep_score = db.Column(db.Integer, nullable=False)
    anx_score = db.Column(db.Integer, nullable=False)
    str_score = db.Column(db.Integer, nullable=False)
    dep_pred = db.Column(db.Integer, nullable=False)
    anx_pred = db.Column(db.Integer, nullable=False)
    str_pred = db.Column(db.Integer, nullable=False)
    dep_proba_json = db.Column(db.Text, nullable=False)
    anx_proba_json = db.Column(db.Text, nullable=False)
    str_proba_json = db.Column(db.Text, nullable=False)
    raw_answers_json = db.Column(db.Text, nullable=True)
    user = db.relationship("User", backref="assessments")

class Review(db.Model):
    __tablename__ = "reviews"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    session_id = db.Column(db.String(64), index=True, nullable=True)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=True)
    rating = db.Column(db.Integer, nullable=False)
    comment = db.Column(db.Text, nullable=True)
    name = db.Column(db.String(120), nullable=True)
    user = db.relationship("User", backref="reviews")

class ShapTopK(db.Model):
    __tablename__ = "shap_topk"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    assessment_id = db.Column(db.String(36), db.ForeignKey("assessments.id"), index=True, nullable=False)
    condition = db.Column(db.String(16), nullable=False)
    predicted_class = db.Column(db.Integer, nullable=False)
    feature_code = db.Column(db.String(8), nullable=False)
    feature_value = db.Column(db.Integer, nullable=True)
    shap_value = db.Column(db.Float, nullable=False)
    abs_shap = db.Column(db.Float, nullable=False)
    assessment = db.relationship("Assessment", backref="shap_items")

class DeleteAllForm(FlaskForm):
    submit = SubmitField("Delete all history")

class DeleteOneForm(FlaskForm):
    submit = SubmitField("Delete")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)

with app.app_context():
    db.create_all()

class SignupForm(FlaskForm):
    name = StringField(_("Name (optional)"), validators=[Length(max=120)])
    email = StringField(_("Email"), validators=[DataRequired(), Email(), Length(max=255)])
    password = PasswordField(_("Password"), validators=[DataRequired(), Length(min=6, max=128)])
    confirm = PasswordField(_("Confirm Password"), validators=[DataRequired(), EqualTo("password", message=_("Passwords must match."))])
    submit = SubmitField(_("Create Account"))

class LoginForm(FlaskForm):
    email = StringField(_("Email"), validators=[DataRequired(), Email(), Length(max=255)])
    password = PasswordField(_("Password"), validators=[DataRequired()])
    submit = SubmitField(_("Log in"))

class ReviewForm(FlaskForm):
    name = StringField(_("Name (optional)"), validators=[Length(max=120)])
    rating = SelectField(
        _("Your rating"),
        choices=[("5","★★★★★"), ("4","★★★★☆"), ("3","★★★☆☆"), ("2","★★☆☆☆"), ("1","★☆☆☆☆")],
        validators=[DataRequired()]
    )
    comment = TextAreaField(_("Comments (optional)"), validators=[Optional(), Length(max=2000)])
    submit = SubmitField(_("Submit review"))

depression_model = pickle.load(open("depression_model.pkl", "rb"))
anxiety_model    = pickle.load(open("anxiety_model.pkl", "rb"))
stress_model     = pickle.load(open("stress_model.pkl", "rb"))

depression_scaler = pickle.load(open("depression_scaler.pkl", "rb"))
anxiety_scaler    = pickle.load(open("anxiety_scaler.pkl", "rb"))
stress_scaler     = pickle.load(open("stress_scaler.pkl", "rb"))

with open("model_columns.json", "r", encoding="utf-8") as f:
    question_columns = json.load(f)["columns"]

dep_cols = ["Q3A","Q5A","Q10A","Q13A","Q16A","Q17A","Q21A","Q24A","Q26A","Q31A","Q34A","Q37A","Q38A","Q42A"]
anx_cols = ["Q2A","Q4A","Q7A","Q9A","Q15A","Q19A","Q20A","Q23A","Q25A","Q28A","Q30A","Q36A","Q40A","Q41A"]
str_cols = ["Q1A","Q6A","Q8A","Q11A","Q12A","Q14A","Q18A","Q22A","Q27A","Q29A","Q32A","Q33A","Q35A","Q39A"]

def dass_categorize(score: int) -> int:
    if score <= 14: return 0
    elif score <= 28: return 1
    else: return 2

def build_no_symptoms_baseline(scaler, question_columns, starts_at_one=True):
    base_raw = np.ones((1, len(question_columns)), dtype=float) if starts_at_one else np.zeros((1, len(question_columns)), dtype=float)
    return scaler.transform(pd.DataFrame(base_raw, columns=question_columns))

dep_baseline = build_no_symptoms_baseline(depression_scaler, question_columns, starts_at_one=INPUT_SCALE_STARTS_AT_ONE)
anx_baseline = build_no_symptoms_baseline(anxiety_scaler,    question_columns, starts_at_one=INPUT_SCALE_STARTS_AT_ONE)
str_baseline = build_no_symptoms_baseline(stress_scaler,     question_columns, starts_at_one=INPUT_SCALE_STARTS_AT_ONE)

dep_explainer = shap.LinearExplainer(depression_model, dep_baseline, feature_perturbation="interventional")
anx_explainer = shap.LinearExplainer(anxiety_model,    anx_baseline, feature_perturbation="interventional")
str_explainer = shap.LinearExplainer(stress_model,     str_baseline, feature_perturbation="interventional")

def top_contributors_filtered(
    explainer, X_scaled_row, feature_names, pred_class, raw_answers,
    feature_whitelist=None, only_nondefault=True, positive_only=True, min_abs=0.02, topk=5
):
    exp = explainer(X_scaled_row)
    vals = np.asarray(exp.values)
    if vals.ndim == 3 and vals.shape[0] == 1:
        class_vals = vals[0, :, pred_class]
    elif vals.ndim == 2:
        class_vals = vals[0, :]
    else:
        raise ValueError(f"Unexpected SHAP shape: {vals.shape}")

    whitelist_idx = None
    if feature_whitelist is not None:
        idx_map = {f: i for i, f in enumerate(feature_names)}
        whitelist_idx = set(idx_map[f] for f in feature_whitelist if f in idx_map)

    items = []
    for i, v in enumerate(class_vals):
        if whitelist_idx is not None and i not in whitelist_idx: continue
        if only_nondefault and int(raw_answers[i]) == 1: continue
        if positive_only and v <= 0: continue
        if abs(v) < min_abs: continue
        items.append((i, abs(v), v))

    if not items: return []
    items.sort(key=lambda t: (-t[1], t[0]))
    items = items[:topk]
    return [(feature_names[i], float(absv), float(signedv)) for i, absv, signedv in items]

def get_locale():
    return session.get("lang", "en")

ADMIN_USERNAME = os.getenv("ADMIN_USER", "admin")
_ADMIN_PLAIN = os.getenv("ADMIN_PASS", "admin123")
ADMIN_PASSWORD_HASH = generate_password_hash(_ADMIN_PLAIN)

class AdminLoginForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired(), Length(max=64)])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=4, max=128)])
    submit = SubmitField("Sign in")

def admin_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("admin_login", next=request.path))
        return view_func(*args, **kwargs)
    return wrapper

class AdminDeleteForm(FlaskForm):
    submit = SubmitField("Delete")

class ReviewFilterForm(FlaskForm):
    date_from = DateField("From (YYYY-MM-DD)", validators=[Opt()])
    date_to   = DateField("To (YYYY-MM-DD)",   validators=[Opt()])
    submit    = SubmitField("Filter")

class ExportForm(FlaskForm):
    date_from = DateField("From (YYYY-MM-DD)", validators=[Opt()])
    date_to   = DateField("To (YYYY-MM-DD)",   validators=[Opt()])
    include_raw = BooleanField("Include raw answers (JSON)")
    submit    = SubmitField("Download CSV")

def _date_floor(d):
    return datetime(d.year, d.month, d.day) if d else None

def _date_ceil(d):
    if not d: return None
    return datetime(d.year, d.month, d.day) + pd.Timedelta(days=1)

@app.route("/lang/<code>")
def set_language(code):
    code = (code or "").lower()
    session["lang"] = code if code in LANGUAGES else "en"
    return redirect(request.referrer or url_for("home"))

I18N_DIR = Path(__file__).parent / "i18n"

LABEL_MAPS = {
    "en": {0: "Normal", 1: "Moderate", 2: "Severe"},
    "si": {0: "සාමාන්‍ය", 1: "මධ්‍යම", 2: "දරුණු"}
}

def load_questions_for_locale(locale_code: str):
    fname = "dass42_questions.si.json" if locale_code == "si" else "dass42_questions.en.json"
    fpath = I18N_DIR / fname
    with open(fpath, "r", encoding="utf-8") as f:
        qmap = json.load(f)
    return {col: qmap.get(col, col) for col in question_columns}

@app.before_request
def _inject_i18n_maps():
    lang = get_locale()
    g.lang = lang
    g.label_map = LABEL_MAPS.get(lang, LABEL_MAPS["en"])
    g.question_map = load_questions_for_locale(lang)

MIN_NONDEFAULT_FOR_CONFIDENCE = 3
SOFTEN_MAX = 0.50
UNIFORM_PRIOR_3 = np.array([1/3, 1/3, 1/3], dtype=float)

def soften_probs_when_limited(probs, evidence_count, min_needed=MIN_NONDEFAULT_FOR_CONFIDENCE):
    probs = np.array(probs, dtype=float)
    if evidence_count <= 0:
        return probs.tolist(), False
    if evidence_count >= min_needed:
        return probs.tolist(), False
    blend = (min_needed - evidence_count) / float(min_needed) * SOFTEN_MAX
    softened = (1.0 - blend) * probs + blend * UNIFORM_PRIOR_3
    softened = softened / softened.sum()
    return softened.tolist(), True

TIPS = {
    "en": {
        "bucket": {
            "Depression": {
                0: ["Your answers are within the normal range. Keep up healthy routines: sleep, exercise, and social connection."],
                1: ["Consider building a daily routine, gentle physical activity, and journaling. If symptoms persist, talk to a professional."],
                2: ["Your result suggests higher symptoms. Please consider contacting a mental health professional for guidance."]
            },
            "Anxiety": {
                0: ["Anxiety items look normal. Practicing short breathing exercises can still be helpful."],
                1: ["Try paced breathing (e.g., 4s inhale/6s exhale), reduce caffeine, and use worry time scheduling."],
                2: ["High anxiety signals. Consider professional support; meanwhile, use grounding techniques (5-4-3-2-1) during spikes."]
            },
            "Stress": {
                0: ["Stress appears normal. Maintain breaks, sleep hygiene, and time boundaries."],
                1: ["Try micro-breaks, task chunking, and saying no to overload. Keep a consistent wind-down routine."],
                2: ["Stress is elevated. Seek support and consider workload changes. Prioritize recovery (sleep, hydration, movement)."]
            }
        },
        "personalizers": {
            "Q9A":  "Frequent relief after stressful situations? Try a post-event cool-down (walk, stretch, 2-minute breathing) to reset.",
            "Q40A": "Worried about situations where you might panic? Rehearse a simple plan and safe person to contact beforehand.",
            "Q25A": "Heartbeat awareness bothers you? Practice slow breathing or box breathing to reduce bodily arousal.",
            "Q42A": "Low initiative? Use the ‘2-minute rule’: start a task for just 2 minutes to break inertia."
        }
    },
    "si": {
        "bucket": {
            "Depression": {
                0: ["ඔබගේ පිළිතුරු සාමාන්‍ය පරාසයේය. නිදාගැනීම, ව්‍යායාම සහ සමාජ සම්බන්ධතා රැකගන්න."],
                1: ["දිනපතා රටාවක්, සුමට ව්‍යායාමයක් සහ ලේඛනගත කිරීම උදව්කරයි. ලක්ෂණ දිගටම පවතින අතර වෘත්තීය උපදෙස් සලකා බලන්න."],
                2: ["උසස් ලක්ෂණ පෙන්වයි. කරුණාකර මානසික සෞඛ්‍ය වෘත්තීයෙකු සමඟ සම්බන්ධ වන්න."]
            },
            "Anxiety": {
                0: ["කළුතැනීම් සාමාන්‍යයකි. කෙටි හුස්ම හුවමාරු පුහුණු කිරීම ද ප්‍රයෝජනවත් වේ."],
                1: ["ආතතික හුස්ම හුවමාරුව (උදා: 4s ඇදගැනීම/6s හමන්න), කැෆීන අඩු කිරීම, ‘කනගාටු කාලය’ සැලසුම් කිරීම උදව් කරයි."],
                2: ["උසස් කළුතැනීම්. වෘත්තීය උපකාරය සලකා බලන්න; අතරතුරේදී 5-4-3-2-1 පදනම් කිරීම භාවිතා කරන්න."]
            },
            "Stress": {
                0: ["ආතතිය සාමාන්‍යයි. විරාම, නිදහස් නිදාගැනීම සහ කාල සීමා පවත්වාගෙන යන්න."],
                1: ["සුළු විරාම, කාර්ය කැබලි කරන ලෙස භ්‍රමණය, වැඩ ඉල්ලීම්වල ‘නැහැ’ කියා ප්‌رතිසංස්කරනය උදව්කරයි."],
                2: ["උසස් ආතතිය. සහාය ලබාගන්න; වැඩ ප්‍රමාණය වෙනස් කිරීම සහ ප්‍රතිසාධනයට (නිදාගැනීම, ජලය, චලනය) ප්‍රමුඛතාව දෙන්න."]
            }
        },
        "personalizers": {
            "Q9A":  "කාර්යයෙන් පසු තාවකාලික සහනයක් දැනේද? අවසන් වූ පසු මිනිත්තු 2ක් හුස්ම හුවමාරුවක්/සෙමින් ඇවිදීම කරන්න.",
            "Q40A": "පැනඟිම ගැන බියද? පෙර සැලැස්මක් සහ උපකාරක පුද්ගලයෙකු හඳුනාගෙන රිහර්සල් කරන්න.",
            "Q25A": "හෘද සීඝ්‍රතාව ගැන අසාත්මිකද? ආතතික හුස්ම හුවමාරුව හෝ ‘box breathing’ පුහුණු කරන්න.",
            "Q42A": "ආරම්භ කිරීම අමාරුද? ‘මිනිත්තු 2 නියමය’ භාවිතයෙන් කාර්යය ආරම්භ කරන්න."
        }
    }
}

def build_advice(lang: str, dep_pred: int, anx_pred: int, str_pred: int,
                 shap_dep_codes=None, shap_anx_codes=None, shap_str_codes=None):
    cfg = TIPS.get(lang, TIPS["en"])
    bucket = cfg["bucket"]
    pers   = cfg["personalizers"]
    tips = []
    tips += bucket["Depression"].get(dep_pred, [])
    tips += bucket["Anxiety"].get(anx_pred, [])
    tips += bucket["Stress"].get(str_pred, [])
    seen = set()
    def add_personalizers(codes):
        for code in (codes or []):
            if code in pers and code not in seen:
                tips.append(pers[code])
                seen.add(code)
    add_personalizers(shap_dep_codes)
    add_personalizers(shap_anx_codes)
    add_personalizers(shap_str_codes)
    return tips

@app.route("/home", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        consent_checked = request.form.get("consent_checked") == "on"
        action = request.form.get("action")
        if not consent_checked:
            return render_template(
                "home.html",
                consent_error=_("Please check consent to continue."),
                consent_version=CONSENT_VERSION,
                consent_retention=CONSENT_RETENTION,
                consent_contact=CONSENT_CONTACT_EMAIL
            )
        session["consent_version"] = CONSENT_VERSION
        session["consent_accepted_at"] = datetime.utcnow().isoformat(timespec="seconds")
        if action == "login":
            return redirect(url_for("login"))
        elif action == "signup":
            return redirect(url_for("signup"))
        elif action == "guest":
            current_session_id()
            session["is_anonymous"] = True
            session.pop("user_id", None)
            return redirect(url_for("assessment"))
        else:
            return render_template(
                "home.html",
                consent_error=_("Unknown action."),
                consent_version=CONSENT_VERSION,
                consent_retention=CONSENT_RETENTION,
                consent_contact=CONSENT_CONTACT_EMAIL
            )
    return render_template(
        "home.html",
        consent_version=CONSENT_VERSION,
        consent_retention=CONSENT_RETENTION,
        consent_contact=CONSENT_CONTACT_EMAIL
    )

@app.route("/signup", methods=["GET", "POST"])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        name = (form.name.data or "").strip()
        if User.query.filter_by(email=email).first():
            return render_template("signup.html", form=form, error=_("An account with this email already exists."))
        user = User(email=email, name=name)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        sid = session.get("anon_id")
        if sid:
            Assessment.query.filter(
                Assessment.session_id == sid,
                Assessment.user_id.is_(None)
            ).update({Assessment.user_id: current_user.id})
            db.session.commit()
        session["is_anonymous"] = False
        return redirect(url_for("user_panel"))
    return render_template("signup.html", form=form)

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(form.password.data):
            return render_template("login.html", form=form, error=_("Invalid email or password."))
        if not user.is_active:
            return render_template("login.html", form=form, error=_("This account is disabled."))
        login_user(user)
        sid = session.get("anon_id")
        if sid:
            Assessment.query.filter(
                Assessment.session_id == sid,
                Assessment.user_id.is_(None)
            ).update({Assessment.user_id: current_user.id})
            db.session.commit()
        session["is_anonymous"] = False
        next_url = request.args.get("next")
        return redirect(next_url or url_for("user_panel"))
    return render_template("login.html", form=form)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))

@app.route("/assessment", methods=["GET", "POST"])
def assessment():
    if request.method == "POST":
        try:
            raw_vals = [int(request.form.get(col, 1)) for col in question_columns]
            vals = raw_vals[:] if INPUT_SCALE_STARTS_AT_ONE else [v - 1 for v in raw_vals]
            X_df = pd.DataFrame([vals], columns=question_columns)
            d_x = depression_scaler.transform(X_df)
            a_x = anxiety_scaler.transform(X_df)
            s_x = stress_scaler.transform(X_df)
            d_pred = int(depression_model.predict(d_x)[0])
            a_pred = int(anxiety_model.predict(a_x)[0])
            s_pred = int(stress_model.predict(s_x)[0])
            d_proba_raw = depression_model.predict_proba(d_x)[0].tolist()
            a_proba_raw = anxiety_model.predict_proba(a_x)[0].tolist()
            s_proba_raw = stress_model.predict_proba(s_x)[0].tolist()
            d_top = top_contributors_filtered(dep_explainer, d_x, question_columns, d_pred, raw_vals,
                                              feature_whitelist=dep_cols, only_nondefault=True, positive_only=True, min_abs=0.02, topk=5)
            a_top = top_contributors_filtered(anx_explainer, a_x, question_columns, a_pred, raw_vals,
                                              feature_whitelist=anx_cols, only_nondefault=True, positive_only=True, min_abs=0.02, topk=5)
            s_top = top_contributors_filtered(str_explainer, s_x, question_columns, s_pred, raw_vals,
                                              feature_whitelist=str_cols, only_nondefault=True, positive_only=True, min_abs=0.02, topk=5)
            ans = dict(zip(question_columns, raw_vals))
            dep_sum = sum(ans[c] for c in dep_cols)
            anx_sum = sum(ans[c] for c in anx_cols)
            str_sum = sum(ans[c] for c in str_cols)
            dep_evid = sum(1 for c in dep_cols if ans[c] > 1)
            anx_evid = sum(1 for c in anx_cols if ans[c] > 1)
            str_evid = sum(1 for c in str_cols if ans[c] > 1)
            d_proba, d_limited = soften_probs_when_limited(d_proba_raw, dep_evid)
            a_proba, a_limited = soften_probs_when_limited(a_proba_raw, anx_evid)
            s_proba, s_limited = soften_probs_when_limited(s_proba_raw, str_evid)
            save_flag = request.form.get("save_result") == "on"
            store_raw = request.form.get("store_raw") == "on"
            if save_flag:
                a = Assessment(
                    session_id=current_session_id(),
                    user_id=(current_user.id if current_user.is_authenticated else None),
                    consent_version=session.get("consent_version"),
                    model_version=MODEL_VERSION,
                    dep_score=int(dep_sum), anx_score=int(anx_sum), str_score=int(str_sum),
                    dep_pred=int(d_pred), anx_pred=int(a_pred), str_pred=int(s_pred),
                    dep_proba_json=json.dumps(d_proba),
                    anx_proba_json=json.dumps(a_proba),
                    str_proba_json=json.dumps(s_proba),
                    raw_answers_json=json.dumps(ans) if store_raw else None
                )
                db.session.add(a)
                db.session.flush()
                def persist_topk(condition_name, pred_cls, items):
                    for code, absv, signedv in items:
                        db.session.add(ShapTopK(
                            assessment_id=a.id,
                            condition=condition_name,
                            predicted_class=pred_cls,
                            feature_code=code,
                            feature_value=int(ans.get(code, 0)),
                            shap_value=float(signedv),
                            abs_shap=float(absv)
                        ))
                persist_topk("Depression", d_pred, d_top)
                persist_topk("Anxiety",    a_pred, a_top)
                persist_topk("Stress",     s_pred, s_top)
                db.session.commit()
            def pretty(items):
                return [{"code": code,
                         "text": g.question_map.get(code, code),
                         "abs": round(absv, 4),
                         "signed": round(signedv, 4)} for code, absv, signedv in items]
            rule_dep_txt = g.label_map[dass_categorize(dep_sum)]
            rule_anx_txt = g.label_map[dass_categorize(anx_sum)]
            rule_str_txt = g.label_map[dass_categorize(str_sum)]
            return render_template(
                "index.html",
                question_map=g.question_map,
                question_columns=question_columns,
                input_values=raw_vals,
                depression=g.label_map[d_pred],
                anxiety=g.label_map[a_pred],
                stress=g.label_map[s_pred],
                d_proba=d_proba, a_proba=a_proba, s_proba=s_proba,
                d_top=pretty(d_top), a_top=pretty(a_top), s_top=pretty(s_top),
                rule_dep=rule_dep_txt, rule_anx=rule_anx_txt, rule_str=rule_str_txt,
                d_limited=d_limited, a_limited=a_limited, s_limited=s_limited
            )
        except Exception as e:
            return f"<pre>Error: {e}</pre>"
    return render_template(
        "index.html",
        question_map=g.question_map,
        question_columns=question_columns,
        input_values=None,
        depression=None, anxiety=None, stress=None
    )

@app.route("/advice/<assessment_id>")
@login_required
def advice_page(assessment_id):
    a = Assessment.query.get_or_404(assessment_id)
    if a.user_id != current_user.id:
        return "<h3>Not authorized.</h3>", 403
    shap_dep = (ShapTopK.query
                .filter_by(assessment_id=assessment_id, condition="Depression")
                .order_by(ShapTopK.abs_shap.desc()).limit(5).all())
    shap_anx = (ShapTopK.query
                .filter_by(assessment_id=assessment_id, condition="Anxiety")
                .order_by(ShapTopK.abs_shap.desc()).limit(5).all())
    shap_str = (ShapTopK.query
                .filter_by(assessment_id=assessment_id, condition="Stress")
                .order_by(ShapTopK.abs_shap.desc()).limit(5).all())
    dep_codes = [r.feature_code for r in shap_dep]
    anx_codes = [r.feature_code for r in shap_anx]
    str_codes = [r.feature_code for r in shap_str]
    tips = build_advice(g.lang, a.dep_pred, a.anx_pred, a.str_pred,
                        dep_codes, anx_codes, str_codes)
    page_title = "Personalized tips" if g.lang == "en" else "පෞද්ගලික උපදෙස්"
    return render_template("advice.html",
                           page_title=page_title,
                           assessment=a,
                           label_map=g.label_map,
                           tips=tips)

@app.route("/history")
@login_required
def history():
    items = (Assessment.query
             .filter(Assessment.user_id == current_user.id)
             .order_by(Assessment.created_at.desc())
             .all())
    delete_all_form = DeleteAllForm()
    delete_one_form = DeleteOneForm()
    return render_template("history.html",
                           items=items,
                           label_map=g.label_map,
                           delete_all_form=delete_all_form,
                           delete_one_form=delete_one_form)

@app.route("/history/<assessment_id>")
@login_required
def history_detail(assessment_id):
    item = Assessment.query.get_or_404(assessment_id)
    if item.user_id != current_user.id:
        return "<h3>Not authorized to view this assessment.</h3>", 403
    shap_dep = (ShapTopK.query
                .filter_by(assessment_id=assessment_id, condition="Depression")
                .order_by(ShapTopK.abs_shap.desc()).limit(5).all())
    shap_anx = (ShapTopK.query
                .filter_by(assessment_id=assessment_id, condition="Anxiety")
                .order_by(ShapTopK.abs_shap.desc()).limit(5).all())
    shap_str = (ShapTopK.query
                .filter_by(assessment_id=assessment_id, condition="Stress")
                .order_by(ShapTopK.abs_shap.desc()).limit(5).all())
    return render_template("history_detail.html",
                           item=item,
                           shap_dep=shap_dep, shap_anx=shap_anx, shap_str=shap_str,
                           label_map=g.label_map)

@app.route("/history/<assessment_id>/delete", methods=["POST"])
@login_required
def history_delete_one(assessment_id):
    item = Assessment.query.get_or_404(assessment_id)
    if item.user_id != current_user.id:
        return "<h3>Not authorized.</h3>", 403
    ShapTopK.query.filter_by(assessment_id=item.id).delete(synchronize_session=False)
    db.session.delete(item)
    db.session.commit()
    return redirect(url_for("history"))

@app.route("/history/delete_all", methods=["POST"])
@login_required
def history_delete_all():
    ids = [a.id for a in Assessment.query
           .filter(Assessment.user_id == current_user.id).all()]
    if ids:
        ShapTopK.query.filter(ShapTopK.assessment_id.in_(ids)).delete(synchronize_session=False)
        Assessment.query.filter(Assessment.id.in_(ids)).delete(synchronize_session=False)
        db.session.commit()
    return redirect(url_for("history"))

@app.route("/reviews", methods=["GET", "POST"])
def reviews():
    form = ReviewForm()
    saved = False
    if form.validate_on_submit():
        r = Review(
            session_id=session.get("anon_id"),
            user_id=(current_user.id if current_user.is_authenticated else None),
            rating=int(form.rating.data),
            comment=(form.comment.data or "").strip() or None,
            name=(form.name.data or "").strip() or None
        )
        db.session.add(r)
        db.session.commit()
        saved = True
    recent = (Review.query
              .order_by(Review.created_at.desc())
              .limit(12)
              .all())
    return render_template(
        "review.html",
        form=form,
        reviews=recent,
        saved=saved
    )

@app.route("/dashboard")
@login_required
def dashboard():
    q = (Assessment.query
         .filter(Assessment.user_id == current_user.id)
         .order_by(Assessment.created_at.asc()))
    rows = q.all()
    if not rows:
        return render_template("dashboard.html", has_data=False)
    dates = [r.created_at.strftime("%Y-%m-%d %H:%M") for r in rows]
    dep_probs = [json.loads(r.dep_proba_json) for r in rows]
    anx_probs = [json.loads(r.anx_proba_json) for r in rows]
    str_probs = [json.loads(r.str_proba_json) for r in rows]
    dep_scores = [r.dep_score for r in rows]
    anx_scores = [r.anx_score for r in rows]
    str_scores = [r.str_score for r in rows]
    dep_p_ms = [p[1] + p[2] for p in dep_probs]
    anx_p_ms = [p[1] + p[2] for p in anx_probs]
    str_p_ms = [p[1] + p[2] for p in str_probs]
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Scatter(x=dates, y=dep_p_ms, mode="lines+markers", name="Depression"))
    fig_prob.add_trace(go.Scatter(x=dates, y=anx_p_ms, mode="lines+markers", name="Anxiety"))
    fig_prob.add_trace(go.Scatter(x=dates, y=str_p_ms, mode="lines+markers", name="Stress"))
    fig_prob.update_layout(
        title="Probability of Moderate+Severe over time",
        xaxis_title="Date",
        yaxis_title="Prob.",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_prob_json = json.dumps(fig_prob, cls=PlotlyJSONEncoder)
    fig_scores = go.Figure()
    fig_scores.add_trace(go.Scatter(x=dates, y=dep_scores, mode="lines+markers", name="Depression sum"))
    fig_scores.add_trace(go.Scatter(x=dates, y=anx_scores, mode="lines+markers", name="Anxiety sum"))
    fig_scores.add_trace(go.Scatter(x=dates, y=str_scores, mode="lines+markers", name="Stress sum"))
    fig_scores.update_layout(
        title="Rule-based subscale sums (DASS-42)",
        xaxis_title="Date",
        yaxis_title="Sum",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_scores_json = json.dumps(fig_scores, cls=PlotlyJSONEncoder)
    COLOR_NORMAL   = "#22c55e"
    COLOR_MODERATE = "#f59e0b"
    COLOR_SEVERE   = "#ef4444"
    def ema_triplets(prob_list, alpha=0.7):
        s = None
        out = []
        for p in prob_list:
            if s is None:
                s = [float(p[0]), float(p[1]), float(p[2])]
            else:
                s = [alpha * float(p[i]) + (1 - alpha) * s[i] for i in range(3)]
            out.append(s)
        return out
    dep_probs_ema = ema_triplets(dep_probs, alpha=0.7)
    anx_probs_ema = ema_triplets(anx_probs, alpha=0.7)
    str_probs_ema = ema_triplets(str_probs, alpha=0.7)
    N = 8
    last_dates = [r.created_at.strftime("%m-%d") for r in rows[-N:]]
    dep_last = dep_probs_ema[-N:]
    anx_last = anx_probs_ema[-N:]
    str_last = str_probs_ema[-N:]
    def stacked_for(condition_name, dates_x, prob_triplets):
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Normal",   x=dates_x, y=[p[0] for p in prob_triplets], marker_color=COLOR_NORMAL))
        fig.add_trace(go.Bar(name="Moderate", x=dates_x, y=[p[1] for p in prob_triplets], marker_color=COLOR_MODERATE))
        fig.add_trace(go.Bar(name="Severe",   x=dates_x, y=[p[2] for p in prob_triplets], marker_color=COLOR_SEVERE))
        fig.update_layout(
            barmode="stack",
            title=f"{condition_name}: recency-weighted probability (last {len(dates_x)}, EMA α=0.7)",
            yaxis=dict(range=[0, 1], title="Prob."),
            xaxis_title="Assessment",
            margin=dict(l=40, r=20, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    fig_dep_stack_json = stacked_for("Depression", last_dates, dep_last)
    fig_anx_stack_json = stacked_for("Anxiety",    last_dates, anx_last)
    fig_str_stack_json = stacked_for("Stress",     last_dates, str_last)
    return render_template(
        "dashboard.html",
        has_data=True,
        fig_prob=fig_prob_json,
        fig_scores=fig_scores_json,
        fig_dep_stack=fig_dep_stack_json,
        fig_anx_stack=fig_anx_stack_json,
        fig_str_stack=fig_str_stack_json
    )

@app.route("/research")
@login_required
def research():
    if not is_admin():
        return "<h3>Not authorized (admin only).</h3>", 403
    rows = (Assessment.query
            .order_by(Assessment.created_at.asc())
            .all())
    if not rows:
        return render_template("research.html", has_data=False)
    df = pd.DataFrame([{
        "created_at": r.created_at,
        "dep_score": r.dep_score,
        "anx_score": r.anx_score,
        "str_score": r.str_score,
        "dep_pred": r.dep_pred,
        "anx_pred": r.anx_pred,
        "str_pred": r.str_pred,
    } for r in rows])
    sns.set_theme(style="darkgrid")
    def save_hist(col, fname, title):
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(title)
        plt.tight_layout()
        outp = STATIC_PLOTS_DIR / fname
        plt.savefig(outp, dpi=130)
        plt.close()
        return f"plots/{fname}"
    dep_hist_rel = save_hist("dep_score", "dep_hist.png", "Distribution of Depression sums")
    anx_hist_rel = save_hist("anx_score", "anx_hist.png", "Distribution of Anxiety sums")
    str_hist_rel = save_hist("str_score", "str_hist.png", "Distribution of Stress sums")
    plt.figure(figsize=(4.8,4.2))
    corr = df[["dep_score","anx_score","str_score"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Correlation (D/A/S sums)")
    plt.tight_layout()
    corr_rel = "plots/corr_heatmap.png"
    plt.savefig(STATIC_PLOTS_DIR / "corr_heatmap.png", dpi=130)
    plt.close()
    return render_template(
        "research.html",
        has_data=True,
        img_dep_hist=url_for("static", filename=dep_hist_rel),
        img_anx_hist=url_for("static", filename=anx_hist_rel),
        img_str_hist=url_for("static", filename=str_hist_rel),
        img_corr=url_for("static", filename=corr_rel)
    )

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if session.get("is_admin"):
        return redirect(url_for("admin_home"))
    form = AdminLoginForm()
    error = None
    if form.validate_on_submit():
        u = (form.username.data or "").strip()
        p = form.password.data or ""
        if u == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, p):
            session["is_admin"] = True
            session["admin_user"] = u
            nxt = request.args.get("next")
            return redirect(nxt or url_for("admin_home"))
        else:
            error = "Invalid admin credentials."
    return render_template("admin_login.html", form=form, error=error)

@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    session.pop("admin_user", None)
    return redirect(url_for("admin_login"))

@app.route("/admin")
@admin_required
def admin_home():
    return render_template("admin_home.html")

@app.route("/admin/users")
@admin_required
def admin_users():
    users = User.query.order_by(User.created_at.desc()).all()
    delete_form = AdminDeleteForm()
    rows = []
    for u in users:
        a_count = Assessment.query.filter_by(user_id=u.id).count()
        r_count = Review.query.filter_by(user_id=u.id).count()
        rows.append((u, a_count, r_count))
    return render_template("admin_users.html", rows=rows, delete_form=delete_form)

@app.route("/admin/users/<user_id>/delete", methods=["POST"])
@admin_required
def admin_user_delete(user_id):
    form = AdminDeleteForm()
    if not form.validate_on_submit():
        return "Bad request", 400
    user = User.query.get_or_404(user_id)
    a_ids = [a.id for a in Assessment.query.filter_by(user_id=user.id).all()]
    if a_ids:
        ShapTopK.query.filter(ShapTopK.assessment_id.in_(a_ids)).delete(synchronize_session=False)
        Assessment.query.filter(Assessment.id.in_(a_ids)).delete(synchronize_session=False)
    Review.query.filter_by(user_id=user.id).delete(synchronize_session=False)
    db.session.delete(user)
    db.session.commit()
    return redirect(url_for("admin_users"))

@app.route("/admin/reviews", methods=["GET", "POST"])
@admin_required
def admin_reviews():
    form = ReviewFilterForm()
    delete_form = AdminDeleteForm()
    q = Review.query
    date_from = None
    date_to = None
    if form.validate_on_submit():
        date_from = _date_floor(form.date_from.data)
        date_to   = _date_ceil(form.date_to.data)
    else:
        try:
            if request.args.get("from"):
                form.date_from.data = datetime.strptime(request.args.get("from"), "%Y-%m-%d").date()
                date_from = _date_floor(form.date_from.data)
            if request.args.get("to"):
                form.date_to.data = datetime.strptime(request.args.get("to"), "%Y-%m-%d").date()
                date_to = _date_ceil(form.date_to.data)
        except Exception:
            pass
    if date_from:
        q = q.filter(Review.created_at >= date_from)
    if date_to:
        q = q.filter(Review.created_at < date_to)
    reviews = q.order_by(Review.created_at.desc()).limit(300).all()
    return render_template("admin_reviews.html",
                           form=form,
                           delete_form=delete_form,
                           reviews=reviews,
                           date_from=date_from,
                           date_to=date_to)

@app.route("/admin/reviews/<int:rid>/delete", methods=["POST"])
@admin_required
def admin_review_delete(rid):
    form = AdminDeleteForm()
    if not form.validate_on_submit():
        return "Bad request", 400
    r = Review.query.get_or_404(rid)
    db.session.delete(r)
    db.session.commit()
    return redirect(url_for("admin_reviews"))

@app.route("/admin/export", methods=["GET"])
def admin_export():
    if not session.get("is_admin"):
        return redirect(url_for("admin_login"))
    date_from_str = (request.args.get("date_from") or "").strip()
    date_to_str   = (request.args.get("date_to") or "").strip()
    limit_str     = (request.args.get("limit") or "200").strip()
    fmt           = (request.args.get("format") or "").lower()
    limit = 200
    try:
        limit = max(1, min(10000, int(limit_str)))
    except:
        limit = 200
    q = Assessment.query
    def parse_ymd(s):
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return None
    d_from = parse_ymd(date_from_str)
    d_to   = parse_ymd(date_to_str)
    if d_from:
        q = q.filter(Assessment.created_at >= d_from)
    if d_to:
        q = q.filter(Assessment.created_at < (d_to + timedelta(days=1)))
    q = q.order_by(Assessment.created_at.desc())
    rows = q.limit(limit).all()
    if fmt == "csv":
        si = StringIO()
        w = csv.writer(si)
        w.writerow([
            "assessment_id", "created_at_utc",
            "user_id", "user_email", "session_id",
            "dep_score", "anx_score", "str_score",
            "dep_pred", "anx_pred", "str_pred",
            "dep_proba_json", "anx_proba_json", "str_proba_json",
            "model_version", "consent_version"
        ])
        for a in rows:
            w.writerow([
                a.id,
                a.created_at.isoformat(timespec="seconds"),
                a.user_id or "",
                (a.user.email if a.user_id and a.user else ""),
                a.session_id or "",
                a.dep_score, a.anx_score, a.str_score,
                a.dep_pred, a.anx_pred, a.str_pred,
                a.dep_proba_json, a.anx_proba_json, a.str_proba_json,
                a.model_version or "",
                a.consent_version or "",
            ])
        output = si.getvalue()
        si.close()
        filename = f"assessments_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    label_map = LABEL_MAPS.get("en", {0: "Normal", 1: "Moderate", 2: "Severe"})
    return render_template(
        "admin_export.html",
        assessments=rows,
        total=len(rows),
        date_from=date_from_str,
        date_to=date_to_str,
        limit=limit,
        label_map=label_map
    )

@app.route("/panel")
@login_required
def user_panel():
    items = (Assessment.query
             .filter(Assessment.user_id == current_user.id)
             .order_by(Assessment.created_at.desc())
             .all())
    total = len(items)
    latest = items[0] if total else None
    recent = items[:6] if total else []
    return render_template(
        "user_panel.html",
        total=total,
        latest=latest,
        recent=recent,
        label_map=g.label_map
    )

if __name__ == "__main__":
    app.run(debug=True)
