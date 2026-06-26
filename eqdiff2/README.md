#  Pumas × Guanacos × Ovelhas

Simulação numérica (método de **Euler**) de uma rede trófica de **1 predador e 2
presas competidoras** na estepe patagônica, inspirada no artigo
[arXiv:2412.02936](https://arxiv.org/html/2412.02936v1).

O pacote tem duas entregas:

| Arquivo | O que é |
|---------|---------|
| **`app.py`** | Simulador **web 3D interativo** (Dash + Plotly). Mostra as equações, deixa você mexer em **todos os parâmetros em tempo real** e visualizar os rebanhos sobre o relevo, a série temporal e o retrato de fase 3D. |
| **`simulador_patagonia.ipynb`** | **Jupyter notebook** didático: explica o problema, a ecologia, deduz cada termo das equações e mostra passo a passo como integrar a EDO pelo método de Euler. |
| `model.py` | O modelo (campo vetorial das EDOs) e o integrador de Euler, reutilizados pelo app e pelo notebook. |

---

## O modelo matemático

```
dP/dt = -m_P·P + e_G·P·G + e_O·P·O                    (pumas)
dG/dt =  r_G·G·(1 - G/K_G) - a_G·P·G - c_GO·G·O        (guanacos)
dO/dt =  r_O·O·(1 - O/K_O) - a_O·P·O - c_OG·O·G        (ovelhas)
```

| Símbolo | Significado |
|---------|-------------|
| `m_P` | mortalidade natural dos pumas |
| `e_G`, `e_O` | eficiência de conversão ao caçar guanacos / ovelhas |
| `r_G`, `r_O` | taxa de crescimento intrínseco de cada presa |
| `K_G`, `K_O` | capacidade de suporte do ambiente |
| `a_G`, `a_O` | taxa de ataque do puma a guanacos / ovelhas |
| `c_GO`, `c_OG` | competição interespecífica entre as presas |

Integração: **Euler explícito**, `y_{n+1} = y_n + Δt·f(y_n)`.

---

## Como rodar

### 1. Instalar dependências

```bash
cd eqdiff2
pip install -r requirements.txt
```

### 2. Simulador web 3D

```bash
python app.py
```

Abra **http://127.0.0.1:8050** no navegador.

- **▶ Iniciar / ⏸ Pausar / ↺ Reiniciar** controlam a animação.
- Os **sliders** mudam os parâmetros **enquanto a simulação roda** — o efeito aparece nos rebanhos na hora.
- O menu **Cenário** carrega presets prontos:
  - *Coexistência* — oscilações amortecidas até o equilíbrio das três espécies;
  - *Dominância das ovelhas* — as ovelhas vencem a competição (achado central do paper);
  - *Caça intensa de pumas* — pumas extintos → guanacos explodem até a capacidade de suporte;
  - *Sem competição* — predador-presa clássico, sem `c_GO`/`c_OG`.

### 3. Notebook didático

```bash
jupyter notebook simulador_patagonia.ipynb
```

---

## Estrutura

```
eqdiff2/
├── model.py                     # EDOs + método de Euler (núcleo)
├── app.py                       # simulador web 3D (Dash/Plotly)
├── simulador_patagonia.ipynb    # notebook explicativo
├── requirements.txt
└── README.md
```

> **Nota.** O artigo original integra um modelo *adimensional* reduzido com
> Runge–Kutta de 2ª ordem. Aqui usamos a forma **dimensional** (cada parâmetro
> com significado ecológico direto) e o método de **Euler**, por pedido didático.
> O notebook compara Euler com `scipy`/RK para discutir erro e estabilidade.
