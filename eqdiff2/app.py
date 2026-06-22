"""
Simulador 3D interativo: Pumas x Guanacos x Ovelhas na Patagonia
================================================================

Aplicacao web (Dash + Plotly) que integra NUMERICAMENTE, pelo METODO DE EULER,
o sistema de EDOs de uma rede trofica de 1 predador e 2 presas competidoras.

    dP/dt = -m_P*P + e_G*P*G + e_O*P*O
    dG/dt =  r_G*G*(1 - G/K_G) - a_G*P*G - c_GO*G*O
    dO/dt =  r_O*O*(1 - O/K_O) - a_O*P*O - c_OG*O*G

Recursos:
  * Cena 3D ilustrativa: cada animal e um marcador sobre o relevo patagonico;
    o tamanho do rebanho cresce/encolhe com a populacao em tempo real.
  * Serie temporal P(t), G(t), O(t).
  * Retrato de fase 3D (trajetoria no espaco G-O-P).
  * Sliders para TODOS os parametros, ajustaveis durante a simulacao.
  * Cenarios pre-definidos (coexistencia, dominancia das ovelhas, caca de pumas...).
  * Equacoes renderizadas em LaTeX.

Como rodar:
    pip install -r requirements.txt
    python app.py
    # abra http://127.0.0.1:8050 no navegador
"""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update, MATCH

from model import Params, euler_step

# ---------------------------------------------------------------------------
# 1) Especificacao dos parametros (id, rotulo, min, max, passo, padrao)
# ---------------------------------------------------------------------------
PARAM_SPECS = [
    # --- Pumas ---
    ("m_P", "m_P  — mortalidade do puma",          0.0, 1.5,   0.01,   0.50),
    ("e_G", "e_G  — conversao (caça guanaco)",      0.0, 0.003, 0.0001, 0.0011),
    ("e_O", "e_O  — conversao (caça ovelha)",       0.0, 0.003, 0.0001, 0.0009),
    # --- Guanacos ---
    ("r_G", "r_G  — crescimento do guanaco",        0.0, 2.0,   0.01,   1.10),
    ("K_G", "K_G  — capacidade p/ guanacos",      100.0, 1500.0, 10.0,  900.0),
    ("a_G", "a_G  — ataque do puma ao guanaco",     0.0, 0.01,  0.0001, 0.0045),
    ("c_GO", "c_GO — competição (ovelha→guanaco)",  0.0, 0.003, 0.0001, 0.0005),
    # --- Ovelhas ---
    ("r_O", "r_O  — crescimento da ovelha",         0.0, 2.0,   0.01,   1.00),
    ("K_O", "K_O  — capacidade p/ ovelhas",       100.0, 1500.0, 10.0,  900.0),
    ("a_O", "a_O  — ataque do puma à ovelha",       0.0, 0.01,  0.0001, 0.0055),
    ("c_OG", "c_OG — competição (guanaco→ovelha)",  0.0, 0.003, 0.0001, 0.0003),
]
PARAM_IDS = [s[0] for s in PARAM_SPECS]

# Condicoes iniciais e controle numerico
IC_SPECS = [
    ("P0", "P₀  — pumas iniciais",     0.0, 400.0, 1.0, 40.0),
    ("G0", "G₀  — guanacos iniciais",  0.0, 1000.0, 1.0, 200.0),
    ("O0", "O₀  — ovelhas iniciais",   0.0, 1000.0, 1.0, 150.0),
]
NUM_SPECS = [
    ("dt", "Δt  — passo de Euler",     0.005, 0.05, 0.005, 0.02),
    ("spf", "passos por quadro",       1, 20, 1, 6),
]

# ---------------------------------------------------------------------------
# 2) Cenarios pre-definidos (sobrescrevem os sliders)
# ---------------------------------------------------------------------------
PRESETS = {
    "coexistencia": {
        "m_P": 0.50, "e_G": 0.0011, "e_O": 0.0009,
        "r_G": 1.10, "K_G": 900.0, "a_G": 0.0045, "c_GO": 0.0005,
        "r_O": 1.00, "K_O": 900.0, "a_O": 0.0055, "c_OG": 0.0003,
        "P0": 40.0, "G0": 200.0, "O0": 150.0, "dt": 0.02, "spf": 6,
    },
    "ovelhas_dominam": {
        "m_P": 0.40, "e_G": 0.0010, "e_O": 0.0011,
        "r_G": 0.80, "K_G": 700.0, "a_G": 0.0050, "c_GO": 0.0012,
        "r_O": 0.70, "K_O": 700.0, "a_O": 0.0035, "c_OG": 0.0003,
        "P0": 40.0, "G0": 250.0, "O0": 150.0, "dt": 0.02, "spf": 6,
    },
    "caca_de_pumas": {
        "m_P": 1.00, "e_G": 0.0006, "e_O": 0.0004,
        "r_G": 1.10, "K_G": 900.0, "a_G": 0.0045, "c_GO": 0.0008,
        "r_O": 1.00, "K_O": 900.0, "a_O": 0.0055, "c_OG": 0.0004,
        "P0": 60.0, "G0": 200.0, "O0": 150.0, "dt": 0.02, "spf": 6,
    },
    "sem_competicao": {
        "m_P": 0.50, "e_G": 0.0011, "e_O": 0.0009,
        "r_G": 1.10, "K_G": 900.0, "a_G": 0.0045, "c_GO": 0.0,
        "r_O": 1.00, "K_O": 900.0, "a_O": 0.0055, "c_OG": 0.0,
        "P0": 40.0, "G0": 200.0, "O0": 150.0, "dt": 0.02, "spf": 6,
    },
}
PRESET_LABELS = {
    "coexistencia": "Coexistência (oscilações amortecidas)",
    "ovelhas_dominam": "Dominância das ovelhas (achado do paper)",
    "caca_de_pumas": "Caça intensa de pumas → guanacos explodem",
    "sem_competicao": "Sem competição entre presas",
}

# ---------------------------------------------------------------------------
# 3) Cenario 3D: relevo da estepe + posicoes fixas de cada rebanho
# ---------------------------------------------------------------------------
GRID = np.linspace(0, 100, 60)
TX, TY = np.meshgrid(GRID, GRID)


def terrain_z(x, y):
    """Relevo analitico (colinas suaves) avaliado em (x, y)."""
    return (7.0 * np.sin(x / 17.0) * np.cos(y / 21.0)
            + 4.0 * np.sin(y / 11.0)
            + 3.0 * np.cos(x / 13.0))


TZ = terrain_z(TX, TY)

# Posicoes fixas (seed fixa => os animais nao "tremem" entre quadros)
MAX_MARK = 220          # n. maximo de marcadores desenhados por especie
MARK_DIV = 4.0          # 1 marcador a cada ~4 individuos
_rng = np.random.default_rng(7)


def _herd_positions(n, x_range, y_range):
    xs = _rng.uniform(*x_range, n)
    ys = _rng.uniform(*y_range, n)
    zs = terrain_z(xs, ys) + 1.5
    return xs, ys, zs


# guanacos (campo todo), ovelhas (concentradas), pumas (espalhados)
GX, GY, GZ = _herd_positions(MAX_MARK, (4, 96), (4, 96))
OX, OY, OZ = _herd_positions(MAX_MARK, (10, 90), (10, 90))
PX, PY, PZ = _herd_positions(MAX_MARK, (2, 98), (2, 98))

SPECIES_STYLE = {
    "G": dict(color="#c98a3a", name="Guanacos", symbol="circle"),
    "O": dict(color="#f2f2f2", name="Ovelhas",  symbol="circle"),
    "P": dict(color="#d6452b", name="Pumas",    symbol="diamond"),
}


def n_markers(pop):
    return int(min(MAX_MARK, round(max(pop, 0.0) / MARK_DIV)))


def build_scene(P, G, O):
    """Cena 3D ilustrativa: relevo + rebanhos dimensionados pela populacao."""
    fig = go.Figure()

    # relevo
    fig.add_trace(go.Surface(
        x=TX, y=TY, z=TZ,
        colorscale=[[0, "#6b5d3f"], [0.5, "#8a7a4e"], [1, "#b6a36a"]],
        showscale=False, opacity=0.95, hoverinfo="skip",
        lighting=dict(ambient=0.6, diffuse=0.8), name="estepe",
    ))

    # rebanhos
    for key, (xs, ys, zs), pop in (
        ("G", (GX, GY, GZ), G),
        ("O", (OX, OY, OZ), O),
        ("P", (PX, PY, PZ), P),
    ):
        k = n_markers(pop)
        st = SPECIES_STYLE[key]
        fig.add_trace(go.Scatter3d(
            x=xs[:k], y=ys[:k], z=zs[:k],
            mode="markers",
            marker=dict(size=6 if key != "P" else 8,
                        color=st["color"], symbol=st["symbol"],
                        line=dict(width=0.5, color="#222")),
            name=f"{st['name']}: {pop:,.0f}",
            hovertemplate=f"{st['name']}: {pop:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="#0e1116",
        font=dict(color="#e8e8e8"),
        legend=dict(x=0.0, y=0.98, bgcolor="rgba(0,0,0,0.35)"),
        title=dict(text="Estepe patagônica — rebanhos em tempo real",
                   x=0.5, font=dict(size=14)),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.25),
            bgcolor="#0e1116",
        ),
        uirevision="scene",   # preserva a rotacao do usuario entre quadros
    )
    return fig


def build_timeseries(t, P, G, O):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=G, name="Guanacos", line=dict(color="#c98a3a", width=2)))
    fig.add_trace(go.Scatter(x=t, y=O, name="Ovelhas",  line=dict(color="#cfd3da", width=2)))
    fig.add_trace(go.Scatter(x=t, y=P, name="Pumas",    line=dict(color="#d6452b", width=2)))
    fig.update_layout(
        margin=dict(l=50, r=10, t=30, b=35),
        paper_bgcolor="#0e1116", plot_bgcolor="#161b22",
        font=dict(color="#e8e8e8"),
        title=dict(text="População ao longo do tempo", x=0.5, font=dict(size=14)),
        xaxis=dict(title="tempo t", gridcolor="#283042"),
        yaxis=dict(title="população", gridcolor="#283042"),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        uirevision="ts",
    )
    return fig


def build_phase(P, G, O):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=G, y=O, z=P, mode="lines",
        line=dict(color=np.arange(len(P)), colorscale="Viridis", width=4),
        name="trajetória",
    ))
    if len(P):
        fig.add_trace(go.Scatter3d(
            x=[G[-1]], y=[O[-1]], z=[P[-1]], mode="markers",
            marker=dict(size=6, color="#ffd23f"), name="estado atual",
        ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="#0e1116", font=dict(color="#e8e8e8"),
        showlegend=False,
        title=dict(text="Retrato de fase  (G, O, P)", x=0.5, font=dict(size=14)),
        scene=dict(
            xaxis=dict(title="Guanacos", gridcolor="#283042"),
            yaxis=dict(title="Ovelhas", gridcolor="#283042"),
            zaxis=dict(title="Pumas", gridcolor="#283042"),
            bgcolor="#0e1116",
        ),
        uirevision="phase",
    )
    return fig


# ---------------------------------------------------------------------------
# 4) Estado da simulacao (guardado num dcc.Store)
# ---------------------------------------------------------------------------
WINDOW = 1600  # n. de pontos mantidos no historico (janela deslizante)


def fresh_state(P0, G0, O0):
    return {"t": [0.0], "P": [P0], "G": [G0], "O": [O0]}


def params_from(values):
    """Constroi Params a partir da lista de valores dos sliders (ordem PARAM_IDS)."""
    return Params(**{pid: float(v) for pid, v in zip(PARAM_IDS, values)})


# ---------------------------------------------------------------------------
# 5) Layout
# ---------------------------------------------------------------------------
EQUATIONS_MD = r"""
$$\frac{dP}{dt} = -\,m_P\,P \;+\; e_G\,P\,G \;+\; e_O\,P\,O$$

$$\frac{dG}{dt} = r_G\,G\!\left(1-\frac{G}{K_G}\right) - a_G\,P\,G - c_{GO}\,G\,O$$

$$\frac{dO}{dt} = r_O\,O\!\left(1-\frac{O}{K_O}\right) - a_O\,P\,O - c_{OG}\,O\,G$$
"""

EULER_MD = r"""
**Método de Euler** (passo a passo):
$$y_{n+1} = y_n + \Delta t \cdot f(y_n)$$
"""


def control_row(spec):
    """Uma linha de controle: rótulo + slider + caixa numérica editável (sincronizados)."""
    pid, label, mn, mx, step, default = spec
    return html.Div([
        html.Label(label, style={"fontSize": "11.5px", "color": "#cdd3dc",
                                 "display": "block", "marginBottom": "1px"}),
        html.Div([
            html.Div(
                dcc.Slider(
                    id={"type": "pslider", "index": pid},
                    min=mn, max=mx, step=step, value=default,
                    marks=None, tooltip=None, updatemode="drag",
                    className="pslider",
                ),
                style={"flex": "1", "minWidth": "70px", "paddingRight": "2px"},
            ),
            dcc.Input(
                id={"type": "pinput", "index": pid},
                type="number", value=default,
                min=mn, max=mx, step=step, debounce=True,
                className="pinput",
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
    ], style={"marginBottom": "7px"})


app = Dash(__name__, title="Patagônia 3D — Pumas, Guanacos e Ovelhas")
server = app.server

CARD = {"background": "#161b22", "borderRadius": "10px",
        "padding": "14px", "marginBottom": "12px",
        "border": "1px solid #283042"}

app.layout = html.Div([
    dcc.Store(id="sim", data=fresh_state(40.0, 200.0, 150.0)),
    dcc.Interval(id="ticker", interval=120, disabled=True),

    # cabecalho
    html.Div([
        html.H2("🐆 Patagônia 3D — Pumas × Guanacos × Ovelhas",
                style={"margin": "0", "color": "#ffd23f"}),
        html.P("Rede trófica de 1 predador e 2 presas competidoras, "
               "integrada numericamente pelo método de Euler "
               "(extensão de Lotka–Volterra · arXiv:2412.02936).",
               style={"margin": "4px 0 0", "color": "#9aa4b2", "fontSize": "13px"}),
    ], style={"padding": "10px 16px"}),

    html.Div([
        # ---------------- COLUNA ESQUERDA: controles ----------------
        html.Div([
            html.Div([
                html.Div([
                    html.Button("▶ Iniciar", id="play", n_clicks=0, className="btn"),
                    html.Button("⏸ Pausar", id="pause", n_clicks=0, className="btn"),
                    html.Button("↺ Reiniciar", id="reset", n_clicks=0, className="btn"),
                ], style={"display": "flex", "gap": "6px", "marginBottom": "10px"}),
                html.Label("Cenário:", style={"fontSize": "12px", "color": "#cdd3dc"}),
                dcc.Dropdown(
                    id="preset",
                    options=[{"label": v, "value": k} for k, v in PRESET_LABELS.items()],
                    value="coexistencia", clearable=False,
                    style={"color": "#111"},
                ),
            ], style=CARD),

            html.Div([
                html.H4("Equações do modelo", style={"marginTop": 0, "color": "#ffd23f"}),
                dcc.Markdown(EQUATIONS_MD, mathjax=True,
                             style={"fontSize": "13px"}),
                dcc.Markdown(EULER_MD, mathjax=True, style={"fontSize": "13px"}),
                html.Div(id="readout", style={"fontFamily": "monospace",
                                              "fontSize": "13px", "color": "#9be29b",
                                              "marginTop": "6px"}),
            ], style=CARD),

            html.Div([
                html.H4("Pumas (predador)", style={"margin": "0 0 8px", "color": "#d6452b"}),
                *[control_row(s) for s in PARAM_SPECS[0:3]],
                html.H4("Guanacos (presa nativa)", style={"margin": "10px 0 8px", "color": "#c98a3a"}),
                *[control_row(s) for s in PARAM_SPECS[3:7]],
                html.H4("Ovelhas (presa introduzida)", style={"margin": "10px 0 8px", "color": "#cfd3da"}),
                *[control_row(s) for s in PARAM_SPECS[7:11]],
            ], style=CARD),

            html.Div([
                html.H4("Condições iniciais e numérico", style={"marginTop": 0, "color": "#ffd23f"}),
                *[control_row(s) for s in IC_SPECS],
                *[control_row(s) for s in NUM_SPECS],
            ], style=CARD),
        ], style={"width": "360px", "flex": "0 0 360px",
                  "maxHeight": "calc(100vh - 90px)", "overflowY": "auto",
                  "paddingRight": "8px"}),

        # ---------------- COLUNA DIREITA: graficos ----------------
        html.Div([
            dcc.Graph(id="scene", style={"height": "440px"},
                      config={"displayModeBar": False}),
            html.Div([
                dcc.Graph(id="timeseries", style={"height": "320px", "flex": "1"},
                          config={"displayModeBar": False}),
                dcc.Graph(id="phase", style={"height": "320px", "flex": "1"},
                          config={"displayModeBar": False}),
            ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
        ], style={"flex": "1", "minWidth": "520px"}),
    ], style={"display": "flex", "gap": "14px", "padding": "0 16px 16px",
              "alignItems": "flex-start"}),
], style={"background": "#0e1116", "minHeight": "100vh",
          "fontFamily": "Segoe UI, system-ui, sans-serif", "color": "#e8e8e8"})


# injeta um pouco de CSS para os botoes
app.index_string = app.index_string.replace(
    "</head>",
    """<style>
        .btn{background:#22303f;color:#e8e8e8;border:1px solid #3a4a60;
             border-radius:6px;padding:7px 10px;cursor:pointer;font-size:13px;}
        .btn:hover{background:#2d3f54;}
        ::-webkit-scrollbar{width:8px;} ::-webkit-scrollbar-thumb{background:#2d3f54;border-radius:4px;}

        /* caixa numérica editável ao lado de cada slider */
        .pinput{width:78px;background:#0e1116;color:#ffd23f;border:1px solid #3a4a60;
                border-radius:5px;padding:4px 6px;font-size:12px;text-align:right;
                font-family:Consolas,monospace;}
        .pinput:focus{outline:none;border-color:#ffd23f;
                      box-shadow:0 0 0 2px rgba(255,210,63,0.18);}
        .pinput::-webkit-inner-spin-button,.pinput::-webkit-outer-spin-button{
                opacity:1;height:20px;}

        /* trilho/alça do slider com a cor do tema */
        .pslider .rc-slider-rail{background:#283042;}
        .pslider .rc-slider-track{background:#ffd23f;}
        .pslider .rc-slider-handle{border-color:#ffd23f;background:#ffd23f;opacity:1;}
        .pslider .rc-slider-handle:hover,
        .pslider .rc-slider-handle:active{border-color:#ffe27a;
                box-shadow:0 0 0 4px rgba(255,210,63,0.2);}
    </style></head>""",
)

# ---------------------------------------------------------------------------
# 6) Callbacks
# ---------------------------------------------------------------------------
def sl(pid):
    return State({"type": "pslider", "index": pid}, "value")


PARAM_STATES = [sl(pid) for pid in PARAM_IDS]
IC_STATES = [sl("P0"), sl("G0"), sl("O0")]
NUM_STATES = [sl("dt"), sl("spf")]

ALL_SPECS = PARAM_SPECS + IC_SPECS + NUM_SPECS


def _num(v, default):
    """Converte para float, tolerando caixa vazia (None)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# (a) Sincroniza cada par slider <-> caixa numerica (ambas as direcoes)
@app.callback(
    Output({"type": "pslider", "index": MATCH}, "value"),
    Output({"type": "pinput", "index": MATCH}, "value"),
    Input({"type": "pslider", "index": MATCH}, "value"),
    Input({"type": "pinput", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def sync_pair(slider_val, input_val):
    trig = ctx.triggered_id
    from_slider = isinstance(trig, dict) and trig.get("type") == "pslider"
    val = slider_val if from_slider else input_val
    if val is None:                       # caixa esvaziada: nao propaga
        return no_update, no_update
    return val, val


# (b) Carregar um cenario -> escreve nos sliders (a sync espelha nas caixas)
@app.callback(
    [Output({"type": "pslider", "index": s[0]}, "value", allow_duplicate=True)
     for s in ALL_SPECS],
    Input("preset", "value"),
    prevent_initial_call=True,
)
def load_preset(name):
    cfg = PRESETS[name]
    return [cfg[s[0]] for s in ALL_SPECS]


# (b) Motor: play / pause / reset / tick  ->  estado + liga/desliga o relogio
@app.callback(
    Output("sim", "data"),
    Output("ticker", "disabled"),
    Input("ticker", "n_intervals"),
    Input("play", "n_clicks"),
    Input("pause", "n_clicks"),
    Input("reset", "n_clicks"),
    State("sim", "data"),
    *PARAM_STATES, *IC_STATES, *NUM_STATES,
    prevent_initial_call=True,
)
def engine(_n, _p, _pa, _r, data, *vals):
    trig = ctx.triggered_id
    n_params = len(PARAM_IDS)
    defaults = [s[5] for s in PARAM_SPECS]
    param_vals = [_num(v, d) for v, d in zip(vals[:n_params], defaults)]
    P0 = _num(vals[n_params], 40.0)
    G0 = _num(vals[n_params + 1], 200.0)
    O0 = _num(vals[n_params + 2], 150.0)
    dt = max(_num(vals[n_params + 3], 0.02), 1e-4)
    spf = max(int(_num(vals[n_params + 4], 6)), 1)

    if trig == "play":
        return no_update, False
    if trig == "pause":
        return no_update, True
    if trig == "reset":
        return fresh_state(P0, G0, O0), True

    # trig == "ticker": avanca a integracao de Euler
    p = params_from(param_vals)
    state = (data["P"][-1], data["G"][-1], data["O"][-1])
    t = data["t"][-1]
    for _ in range(spf):
        state = euler_step(state, p, dt)
        t += dt

    for key, val in zip("PGO", state):
        data[key].append(val)
    data["t"].append(t)

    # janela deslizante
    if len(data["t"]) > WINDOW:
        for key in ("t", "P", "G", "O"):
            data[key] = data[key][-WINDOW:]
    return data, no_update


# (c) Redesenha os tres graficos quando o estado muda
@app.callback(
    Output("scene", "figure"),
    Output("timeseries", "figure"),
    Output("phase", "figure"),
    Output("readout", "children"),
    Input("sim", "data"),
)
def redraw(data):
    t = np.array(data["t"]); P = np.array(data["P"])
    G = np.array(data["G"]); O = np.array(data["O"])
    scene = build_scene(P[-1], G[-1], O[-1])
    ts = build_timeseries(t, P, G, O)
    ph = build_phase(P, G, O)
    readout = (f"t = {t[-1]:7.2f}   |   "
               f"P = {P[-1]:6.1f}   G = {G[-1]:6.1f}   O = {O[-1]:6.1f}")
    return scene, ts, ph, readout


if __name__ == "__main__":
    app.run(debug=False, port=8050)
