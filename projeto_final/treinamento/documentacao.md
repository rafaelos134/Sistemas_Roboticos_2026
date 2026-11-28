# Sumô e futebol com aprendizado por reforço (PPO)

Este documento descreve como a política de sumô foi treinada com PPO, como ela é avaliada e como roda no
psr-hero-kit. O treino segue um **currículo em três fases**:

| Fase | Adversário durante o treino | Arquivo exportado |
|---|---|---|
| 1 | parado (`robot1:=none`) | `rl/exports/sumo/sumo_rl_v1.py` |
| 2 | 5 adversários com script, com ganhos sorteados a cada episódio | `rl/exports/sumo/sumo_rl_v2.py` |
| 3 | *self-play*: cópias congeladas da própria política + os scripts | `rl/exports/sumo/sumo_rl_v3.py` = **`sumo_rl.py`** (versão final) |

As seções 2 a 4 descrevem a base comum (ambiente, observação, recompensa, PPO) e os resultados da fase 1. As seções 5 e 6
descrevem as fases 2 e 3, e a seção 7 traz a avaliação final, os vídeos e o teleop.

> Notação: o robô que aprende é o `hero_0`, e o adversário é o `hero_1`, o segundo robô do kit (os robôs são numerados
> a partir de 0). Todos os arquivos funcionam dos dois lados, e os testes foram feitos nos dois.

## 1. Visão geral

```
 treino (GPU, ~200 mil passos/s)          exportação                  execução (kit, ROS 1)
 ┌────────────────────────────┐   ┌───────────────────────────┐   ┌───────────────────────────┐
 │ rl/sumo_env.py             │   │ rl/export_sumo.py         │   │ rl/exports/sumo/          │
 │ 2048 ringues em paralelo   │──►│ pesos do actor + norma    │──►│   sumo_rl.py              │
 │ no Genesis (mesma física   │   │ da observação embutidos   │   │ rospy + numpy             │
 │ do psr_sim)                │   │ em base64 no .py          │   │ /hero_N/odom → cmd_vel    │
 │ rl/ppo.py  (actor+critic)  │   └───────────────────────────┘   └───────────────────────────┘
 └────────────────────────────┘                                          ▲
                                       rl/eval_sumo.py: roda o mesmo .py no psr_sim real
                                       (ringue elevado, queda, juiz), sem ROS, ou na janela do VNC
```

**Por que os pesos ficam dentro do `.py` e não em ONNX ou `.pt`?** A Seção 3 do enunciado exige que cada jogo
seja **um único arquivo** que use **apenas Python padrão, numpy, rospy, as mensagens do container e
`psr_hero.utils`**. Então o `onnxruntime`, o `torch` e um arquivo de pesos separado ficam de fora. A rede do actor é pequena
(cerca de 5 mil parâmetros) e roda com `numpy` em microssegundos. O arquivo gerado tem cerca de 34 kB e funciona no
container ROS do kit (Python 3.8, numpy 1.17).

## 2. Ambiente de treino (`rl/sumo_env.py`)

### 2.1 Física

O ambiente usa o **Genesis** com `scene.build(n_envs=2048)`: são 2048 ringues independentes simulados juntos na GPU.
O URDF do HeRo e todas as constantes físicas são **importados de `sim/psr_sim/world.py`**, então não há cópia
que possa ficar desatualizada:

| Parâmetro | Valor | Origem |
|---|---|---|
| passo de física `DT` | 0,01 s | `world.py` |
| massa do robô | 0,15 kg | `world.py` |
| força máxima de tração | 0,20 N por eixo | `world.py` (`FORCE_MAX`) |
| ganho de velocidade `KV_LIN`, `KV_YAW` | 3,0 e 0,003 | `world.py` |
| atrito entre os robôs | 0,1 | `world.py` |
| raio do ringue | 0,40 m | `scenarios/sumo.py` |
| duração máxima da rodada | 40 s | `scenarios/sumo.py` |

**Simplificação:** a queda do ringue elevado não é simulada. O episódio termina no instante em que o **centro**
de um robô passa de 0,40 m, que é exatamente a regra do juiz do `psr_sim`.

**Verificação (`rl/check_parity.py`):** um controlador simples, que só gira na direção do adversário e acelera, leva em
média **3,76 s** para empurrar o robô parado para fora no ambiente vetorizado (4096 ringues). No `psr_sim` original, o mesmo
controlador leva **3,76 s**, o que confirma que a física é a mesma.

### 2.2 Adversário

Na fase 1, o `hero_1` recebe velocidade alvo zero em todos os passos. É o mesmo comportamento do `psr_sim` quando um robô não recebe
`/cmd_vel` (`robot1:=none`): ele não anda, mas resiste ao empurrão com a força máxima de 0,20 N.

### 2.3 Episódio e posição inicial

- **Frequência de controle:** 20 Hz (`control_dt = 0,05 s`, 5 passos de física por ação), a mesma do nó ROS.
- **Posição inicial:** 70% dos episódios começam como no `psr_sim`, com o `hero_0` em (−0,18; U(−0,06; 0,06)) e o `hero_1` em
  (0,18; U(−0,06; 0,06)), frente a frente, mais um ruído de ±0,3 rad no ângulo. Os outros 30% começam com os dois robôs em
  posições e ângulos aleatórios dentro de um raio de 0,30 m. Assim a política aprende a se recuperar perto da borda e não
  decora uma única trajetória.
- **Fim do episódio:** um dos robôs sai do ringue (fim terminal) ou passam 40 s (truncamento: o critic estima o
  valor do estado final).
- **Ruído na observação:** 2 mm de desvio-padrão nas posições.

### 2.4 Observação (12 valores, `rl/sumo_obs.py`)

A observação é calculada **no referencial do próprio robô**, ou seja, girada por −yaw. Por isso a mesma política serve
para o `hero_0` e para o `hero_1`, e não depende da orientação do ringue. Tudo vem só do `/hero_N/odom` dos dois robôs.

| # | Valor | Escala |
|---|---|---|
| 0–1 | minha posição em relação ao centro do ringue, no meu referencial | ÷ 0,40 m |
| 2–3 | posição do adversário em relação a mim, no meu referencial | ÷ 0,40 m |
| 4–5 | posição do adversário em relação ao centro, no meu referencial | ÷ 0,40 m |
| 6–7 | cos e sen de (yaw do adversário − meu yaw) | — |
| 8 | minha velocidade para a frente (`twist.linear.x`) | ÷ 0,20 m/s |
| 9 | velocidade para a frente do adversário | ÷ 0,20 m/s |
| 10–11 | ação anterior (linear, angular) | já em [−1, 1] |

A mesma função existe em **torch** (`observation_torch`, usada no treino) e em **numpy** (`observation_np`). O
exportador **copia o texto** de `observation_np` para dentro do `sumo_rl.py`, então o robô usa exatamente a
mesma definição.

### 2.5 Ação

Ação contínua `a = (a_lin, a_ang) ∈ [−1, 1]²` (os valores fora do intervalo são cortados), convertida para
`linear.x = a_lin · 0,20 m/s` e `angular.z = a_ang · 4,0 rad/s`, os limites do simulador (`V_MAX`, `W_MAX`).

### 2.6 Recompensa

A **recompensa base** segue o pedido: vitória ganha, ficar parado perde. Acrescentei o termo de sair do ringue, que é a
regra do jogo; sem ele, sair sozinho não custaria nada.

| Termo | Valor | Quando |
|---|---|---|
| `win` | **+10** | o centro do adversário saiu do ringue (e o meu não) |
| `still` | **−0,05 por passo** | minha velocidade \|v\| < 0,02 m/s (robô parado) |
| `self_out` | **−10** | o meu centro saiu do ringue (e o do adversário não) |

Também há **dois termos de *shaping* baseados em potencial** (Ng, Harada e Russell, 1999):

```
Φ(s) = −1,0 · distância(eu, adversário) + 2,0 · distância(adversário, centro do ringue)
r_shaping = γ · Φ(s') − Φ(s)        (Φ = 0 no estado terminal)
```

Esse tipo de *shaping* **não muda qual é a política ótima** (o teorema de Ng et al. garante isso). Ele só dá um sinal
denso: aproximar-se do adversário e empurrá-lo para a borda passam a render recompensa a cada passo. Os pesos ficam em `RewardConfig`, e a opção
`--no-shaping` treina só com a recompensa base. A comparação dos dois casos está na Seção 4.

## 3. Algoritmo: PPO (`rl/ppo.py`)

PPO (*Proximal Policy Optimization*, Schulman et al., 2017) do tipo *actor-critic*, escrito em PyTorch puro (cerca de 150 linhas)
para deixar cada parte explícita.

### 3.1 Actor (política)

```
obs (12) ─► normalização ─► Linear(12→64) ─► tanh ─► Linear(64→64) ─► tanh ─► Linear(64→2) = μ(s)
                                                                       log σ: 2 parâmetros livres
π(a|s) = Normal(μ(s), diag(σ²))
```

- É uma **política gaussiana diagonal**: a rede dá a média e o desvio-padrão é um parâmetro aprendido que não depende do
  estado. Ele começa em σ = 0,6 e termina perto de σ ≈ 0,11 (linear) e 0,05 (angular), ou seja, a política fica quase
  determinística.
- Na **execução** (no `sumo_rl.py`) usa-se só a **média** μ(s), cortada para [−1, 1], sem sorteio.
- 5 122 pesos e vieses, mais 2 parâmetros de σ.
- Inicialização ortogonal (ganho √2 nas camadas ocultas e 0,01 na saída, para começar com ações próximas de zero).

### 3.2 Critic (função de valor)

```
obs (12) ─► normalização ─► Linear(12→64) ─► tanh ─► Linear(64→64) ─► tanh ─► Linear(64→1) = V(s)
```

- Estima o retorno esperado a partir de cada estado. É usado para calcular as vantagens (GAE) e para estimar o valor do estado
  final quando o episódio é truncado aos 40 s.
- É uma **rede separada**, sem compartilhar pesos com o actor. Tem 5 057 parâmetros e **só é usada no treino**: o
  arquivo exportado leva apenas o actor.

### 3.3 Normalização da observação

Média e variância móveis de cada entrada, atualizadas durante a coleta. A entrada é `(obs − média)/desvio`, cortada em ±5.
As estatísticas finais são exportadas junto com o actor.

### 3.4 Hiperparâmetros

| Hiperparâmetro | Valor |
|---|---|
| ambientes paralelos | 2048 |
| passos por ambiente por iteração | 32 (65 536 amostras por iteração) |
| épocas por iteração / minibatches | 5 / 4 (minibatch de 16 384) |
| γ (desconto) / λ (GAE) | 0,99 / 0,95 |
| *clip* do PPO (ε) | 0,2 |
| coeficiente do valor / da entropia | 1,0 / 0,003 |
| otimizador | Adam, lr inicial 3·10⁻⁴, adaptativo pela divergência KL (alvo 0,02; limites 10⁻⁵ e 10⁻³) |
| norma máxima do gradiente | 1,0 |
| iterações | 500 (32,8 milhões de passos) |

A perda minimizada é `L = L_clip(actor) + 1,0 · L_valor(critic) − 0,003 · entropia`, com
`L_clip = −E[min(ρ·Â, clip(ρ, 1−ε, 1+ε)·Â)]` e `ρ = π_novo(a|s) / π_antigo(a|s)`. As vantagens Â são normalizadas em cada iteração.

## 4. Resultados da fase 1

### 4.1 Treino

![curvas de treino](img/sumo_treino.png)

| Execução | Primeira iteração com ≥ 95% de vitórias (≥ 200 episódios) | Final (iteração 500) | Tempo |
|---|---|---|---|
| `sumo_ppo_v1`: base + *shaping* | iteração 38 (**2,5 milhões de passos**) | 100% de vitórias, 3,78 s por episódio | 2,6 min |
| `sumo_ppo_noshaping`: só a base | iteração 321 (**21,0 milhões de passos**) | 100% de vitórias, 3,99 s por episódio | 2,4 min |

Sem *shaping*, a política passa cerca de 13 milhões de passos **andando sem empurrar**. O episódio dura os 40 s e termina
empatado: ela aprendeu a fugir da penalidade de ficar parada e a não cair, mas a vitória (+10) é rara demais para ser
encontrada por acaso. Com *shaping*, a política aprende a empurrar **cerca de 8 vezes mais rápido**. As duas acabam no mesmo
comportamento, o que confirma que o *shaping* acelera o aprendizado sem mudar a solução.

Vazão: cerca de 200 mil passos de controle por segundo na RTX 5070 Ti, contra cerca de 950 do `psr_sim` com um ringue só.

### 4.2 Avaliação no `psr_sim` real (`rl/eval_sumo.py`)

O arquivo exportado `rl/exports/sumo/sumo_rl.py` roda no simulador original, com ringue elevado, gravidade, queda e o
juiz do jogo, contra um robô parado, 20 rodadas de cada lado. Os registros por rodada estão em `rl/results/*.csv`.

| Política | Lado | Vitórias | Derrotas | Empates | Tempo médio até vencer |
|---|---|---|---|---|---|
| `sumo_ppo_v1` (o `sumo_rl.py` entregue) | `hero_0` | **20/20** | 0 | 0 | 3,77 s |
| `sumo_ppo_v1` | `hero_1` | **20/20** | 0 | 0 | 3,76 s |
| `sumo_ppo_noshaping` | `hero_0` | 20/20 | 0 | 0 | 3,87 s |
| `sumo_ppo_noshaping` | `hero_1` | 20/20 | 0 | 0 | 3,76 s |

### 4.3 Teste no psr-hero-kit (ROS 1 Noetic)

O `sumo_rl.py` foi executado no **container `psr-kit-ros` do kit**, sem alterações, com `roslaunch psr_exercicios sumo.launch`,
contra o simulador do kit. Em cerca de 75 s de cada lado:

| Comando | Placar |
|---|---|
| `robot0:=sumo_rl.py robot1:=none` | `hero_0` 12 × 0, sem empates |
| `robot0:=none robot1:=sumo_rl.py` | `hero_1` 12 × 0, sem empates |

### 4.4 Comportamento aprendido

A política gira até ficar de frente para o adversário, acelera até a velocidade máxima e o empurra em linha reta. Como
ele começa no centro do ringue, a borda mais próxima fica na direção do empurrão. A política ainda não aprendeu
a escolher o ângulo de ataque, porque contra um robô parado o empurrão frontal já vence em cerca de 3,8 s.


## 5. Fase 2: adversários com script

### 5.1 Por que

A política da fase 1 só sabe empurrar de frente um robô parado. Medida contra adversários que se mexem (1024 rodadas
por adversário, ambiente vetorizado), ela **perde 64% das rodadas contra um perseguidor** e **empata quase sempre**
contra um robô que se defende ou foge:

| Adversário | Vitória | Derrota | Empate |
|---|---|---|---|
| parado | 100% | 0% | 0% |
| perseguidor | 17% | **64%** | 19% |
| perseguidor cuidadoso | 20% | 0% | 80% |
| esquiva | 2,5% | 0% | 97,5% |
| aleatório | 95% | 1% | 4% |

### 5.2 Adversários (`rl/opponents.py`)

São controladores determinísticos escritos em torch, para rodarem em lote na GPU junto com a física. **A cada episódio** o
tipo é sorteado e os ganhos também (Kw ∈ [2, 6] rad/s por rad, limiar de alinhamento ∈ [0,3; 0,8] rad, velocidade ∈
[60%, 100%] de V_MAX, raio de guarda da borda ∈ [0,24; 0,32] m). Assim, a política não decora um único adversário.

| Tipo | Peso no sorteio | Comportamento |
|---|---|---|
| `stopped` (parado) | 10% | só segura a posição (mantido para não esquecer a fase 1) |
| `chaser` (perseguidor) | 30% | gira na direção do agente e avança quando está alinhado. É o que a maioria das equipes deve fazer |
| `careful` (cuidadoso) | 25% | perseguidor que volta para o centro quando passa do raio de guarda |
| `evader` (esquiva) | 20% | anda em círculo perto do centro. Quando o agente chega perto, foge de lado (60° em relação à linha entre os dois) |
| `random` (aleatório) | 15% | velocidades aleatórias mantidas por cerca de 1 s. Perto da borda, volta para o centro |

O **`sumo_flanker.py` fica de fora do treino de propósito**. Ele empurra o adversário de lado, na direção da borda mais
próxima dele (a dica do README), e serve só para avaliar a política contra uma estratégia que ela nunca viu.

### 5.3 Mudanças na recompensa

| Termo | Fase 1 | Fases 2 e 3 | Motivo |
|---|---|---|---|
| `draw` (empate: 40 s ou os dois caem) | 0 | **−3** | contra um adversário que resiste, ficar travado de frente até o fim dos 40 s virava uma saída "segura". Com a penalidade, empatar passa a custar |
| *shaping* `shape_center` | 0 | **1,0** | novo potencial `−1,0 · distância(eu, centro)`: estar mais perto do centro do que o adversário é a posição forte no sumô |
| fim aos 40 s | truncamento (o critic estima o valor do estado final) | **terminal** | com a penalidade, o fim da rodada faz parte do jogo |

O potencial completo fica `Φ = −1,0·d(eu, adv) + 2,0·d(adv, centro) − 1,0·d(eu, centro)`. Os demais termos
(`win` +10, `self_out` −10, `still` −0,05) não mudam.

### 5.4 Treino

O treino começa dos pesos da fase 1 (`--init runs/sumo_ppo_v1/model.pt`). O desvio-padrão da política volta a 0,35 para
ela voltar a explorar, porque no fim da fase 1 já estava quase determinística (0,11 e 0,05). São 800 iterações
(**52 milhões de passos, 6,3 min**), com os mesmos hiperparâmetros da Seção 3.4.

![curvas da fase 2](img/sumo_fase2.png)

O perseguidor era o adversário mais difícil: a vitória contra ele sobe de 17% para cerca de 88%, e **as derrotas vão a 0**
nos primeiros 5 milhões de passos. Os empates que sobram (cerca de 12%) acontecem quando os dois ficam frente a frente
com a mesma força.

## 6. Fase 3: self-play com cópias congeladas

### 6.1 Por que

A política da fase 2 vence todos os scripts, mas **contra uma cópia de si mesma empata 88% das rodadas**: as duas vão de
frente e travam. Na competição, os adversários são estratégias de outras equipes, que tendem a ser mais variadas que os
scripts.

### 6.2 Como (`train_sumo.py --phase 3`, `opponents.PolicyPool`)

Só **uma** rede aprende. O adversário é uma **cópia congelada**, cujos pesos não mudam durante o treino. Isso evita os dois
problemas de treinar dois modelos ao mesmo tempo: o alvo que se move a cada iteração e os ciclos (A vence B, B se adapta,
A esquece).

- **Grupo de cópias (*pool*):** começa com a política da fase 2 (posição 0, mantida até o fim). **A cada 100 iterações**
  uma cópia da política atual entra no grupo (máximo de 16; quando enche, sai a mais antiga, menos a da posição 0). Ao
  final eram 16 cópias, salvas em `rl/runs/sumo_ppo_v3/pool/`.
- **"Mais recente":** uma posição do grupo que recebe os pesos atuais a cada iteração, para a política também treinar
  contra o estado atual dela.
- **Sorteio do adversário a cada episódio:**

| Adversário | Peso |
|---|---|
| uma cópia antiga do grupo (sorteio uniforme) | 50% |
| a cópia mais recente | 20% |
| scripts (parado 3%, perseguidor 10%, cuidadoso 8%, esquiva 5%, aleatório 4%) | 30% (para não esquecer como vencê-los) |

- **Simetria:** a observação está no referencial do próprio robô, então a cópia que controla o `hero_1` usa exatamente a
  mesma função `observation_torch`, com os papéis trocados. A cópia usa a **média** da gaussiana, sem sortear, como no
  robô.
- O treino começa da política da fase 2, com desvio-padrão 0,35, e roda 1500 iterações (**98 milhões de passos,
  12,8 min**).

![curvas da fase 3](img/sumo_fase3.png)

A vitória contra as cópias **não sobe de forma contínua, e isso é esperado**: a cada 100 iterações entra uma
cópia mais forte, então o adversário também melhora. A curva cai quando aparece uma cópia que a política ainda não sabe
vencer, e sobe quando ela aprende. A vitória contra os scripts fica entre 80% e 100% durante todo o treino.

### 6.3 Escolha da versão final (`rl/tournament.py`)

Como o desempenho varia entre as cópias, a versão final foi escolhida num torneio. Cada candidata joga 1024 rodadas contra
um painel fixo (os 5 scripts, as políticas das fases 1 e 2 e as cópias das iterações 400, 700, 1000 e 1300), e a
pontuação é a média de (vitória% − derrota%):

| Candidata | fase 2 | it0500 | it0700 | it0900 | it1000 | it1200 | it1400 | **it1500** |
|---|---|---|---|---|---|---|---|---|
| pontuação | 11,5 | 37,7 | 54,6 | 66,5 | 73,4 | 71,3 | 79,6 | **80,6** |

A **última iteração (1500)** ficou em primeiro e foi exportada como `sumo_rl_v3.py` e `sumo_rl.py`.

## 7. Avaliação final no `psr_sim`, vídeos e teleop

### 7.1 Avaliação no `psr_sim` real (20 rodadas, dos dois lados)

Avaliação com o `rl/eval_sumo.py`: os **arquivos exportados** jogam no simulador original (ringue elevado, queda, juiz),
como no kit. Cada linha tem 20 rodadas como `hero_0` e 20 como `hero_1`. Os registros por rodada estão em `rl/results/`.

| Política | Adversário | Como `hero_0` (V/D/E) | Como `hero_1` (V/D/E) | Tempo médio até vencer |
|---|---|---|---|---|
| **fase 3** | parado | **20/0/0** | **20/0/0** | 4,1 s |
| **fase 3** | `sumo_chaser.py` | **20/0/0** | **20/0/0** | 10,6 s |
| **fase 3** | `sumo_flanker.py` (nunca visto no treino) | **20/0/0** | **19/0/1** | 12,3 s |
| **fase 3** | fase 1 | **20/0/0** | **20/0/0** | 13,5 s |
| **fase 3** | fase 2 | **20/0/0** | **20/0/0** | 11,5 s |
| fase 3 | fase 3 (ela mesma) | 10/9/1 | 9/10/1 | 17,0 s |
| fase 2 | `sumo_chaser.py` | 20/0/0 | 20/0/0 | 10,3 s |
| fase 2 | `sumo_flanker.py` | 20/0/0 | 20/0/0 | 12,2 s |
| fase 2 | fase 1 | 2/2/16 | 0/3/17 | — |
| fase 1 | `sumo_flanker.py` | 1/0/19 | 2/0/18 | — |

V/D/E = vitórias, derrotas e empates.

- **A fase 3 vence todos os adversários dos dois lados**, inclusive o `sumo_flanker.py`, que não entrou no treino (39
  vitórias e 1 empate em 40 rodadas), e as versões anteriores (80/80).
- **No kit** (ROS 1, física na CPU, com atraso de rede), contra o flanker: 6 vitórias, 0 derrotas e 2 empates em 8
  rodadas (item 5 da Seção 10).
- Contra ela mesma, os resultados se equilibram (19 × 19 e 2 empates). Na fase 2, o mesmo confronto era 88% de empates. A
  política aprendeu a sair do travamento frontal.
- A fase 1 empata quase tudo contra adversários que se defendem, e a fase 2 empata com a fase 1. Só a fase 3 vence as
  duas.

### 7.2 Comportamento aprendido (observado nos vídeos contra o `sumo_chaser.py`)

- **Fase 1:** gira de frente e empurra em linha reta. Contra quem empurra de volta, os dois travam de frente até o
  fim dos 40 s (`fase1_vs_chaser.mp4`: 3 empates).
- **Fase 2:** não aceita o choque frontal. Passa pelo lado do perseguidor, fica atrás dele e o empurra em linha reta
  até a borda oposta, por cerca de 10 s (`fase2_vs_chaser.mp4`).
- **Fase 3:** desvia para o lado já no primeiro contato e empurra o adversário **de lado**, na direção da borda mais
  próxima dele, sem precisar contorná-lo por completo (`fase3_vs_chaser.mp4`). É a estratégia que o README sugere
  (ângulo de ataque e borda mais próxima) e que o `sumo_flanker.py` implementa à mão, só que aprendida.

Os quadros abaixo foram tirados a cada 1 s da primeira rodada (azul = política, laranja = `sumo_chaser.py`):

![fase 2 contra o chaser](img/quadros_fase2_vs_chaser.png)
![fase 3 contra o chaser](img/quadros_fase3_vs_chaser.png)

### 7.3 Vídeos (`docs/videos/`, gerados com `rl/make_videos.sh`)

São 3 rodadas por vídeo, renderizadas fora da tela por uma câmera do Genesis no `psr_sim` real (30 quadros por segundo,
em tempo real), com placar, mensagem do juiz e relógio da rodada sobrepostos. O azul é sempre a política da primeira coluna.

| Vídeo | Placar (3 rodadas) |
|---|---|
| `fase1_vs_parado.mp4` | 3 × 0 |
| `fase1_vs_chaser.mp4` | 0 × 0, 3 empates |
| `fase1_vs_flanker.mp4` | 0 × 0, 3 empates |
| `fase2_vs_parado.mp4` | 3 × 0 |
| `fase2_vs_chaser.mp4` | 3 × 0 |
| `fase2_vs_flanker.mp4` | 3 × 0 |
| `fase2_vs_fase1.mp4` | 0 × 0, 3 empates |
| `fase3_vs_parado.mp4` | 3 × 0 |
| `fase3_vs_chaser.mp4` | 3 × 0 |
| `fase3_vs_flanker.mp4` | 3 × 0 |
| `fase3_vs_fase1.mp4` | 3 × 0 |
| `fase3_vs_fase2.mp4` | 3 × 0 |
| `fase3_vs_fase3.mp4` | 2 × 1 |

Para gravar um vídeo específico:
`$PY record_video.py exports/sumo/sumo_rl_v3.py --opponent exports/sumo/sumo_flanker.py --rounds 3 --title "..." --out docs/videos/x.mp4`

### 7.4 Teleop: jogar contra as políticas (`rl/teleop_sumo.py`)

Num terminal **dentro do VNC** (http://localhost:3080):

```bash
cd ~/ros2_ws/rl
DISPLAY=:2 ~/ros2_ws/.venv/bin/python teleop_sumo.py --fase 3          # você é o hero_0 (azul)
DISPLAY=:2 ~/ros2_ws/.venv/bin/python teleop_sumo.py --fase 1 --side 1 # você é o hero_1 (laranja)
DISPLAY=:2 ~/ros2_ws/.venv/bin/python teleop_sumo.py --opponent exports/sumo/sumo_flanker.py
```

| Tecla | Ação |
|---|---|
| `w` / `s` ou ↑ / ↓ | velocidade +/− 0,05 m/s (máximo 0,20) |
| `a` / `d` ou ← / → | gira para a esquerda / direita enquanto estiver pressionada |
| espaço | para |
| `r` | zera o placar |
| `q` | sai e mostra o placar |

As teclas são lidas no **terminal**, que precisa estar em foco. A partida aparece na janela do Genesis.

## 8. Como rodar

Tudo no container `ros2_humble_tp_final`, no diretório `~/ros2_ws/rl`, com `PY=~/ros2_ws/.venv/bin/python`.

```bash
# --- treino (currículo) ---
$PY train_sumo.py --phase 1 --iters 500  --name sumo_ppo_v1                                   # ~3 min
$PY train_sumo.py --phase 2 --iters 800  --name sumo_ppo_v2 --init runs/sumo_ppo_v1/model.pt # ~6 min
$PY train_sumo.py --phase 3 --iters 1500 --name sumo_ppo_v3 --init runs/sumo_ppo_v2/model.pt # ~13 min
$PY train_sumo.py --phase 1 --iters 500  --name sumo_ppo_noshaping --no-shaping               # comparação

# --- acompanhar ao vivo: TensorBoard ---
nohup ~/ros2_ws/.venv/bin/tensorboard --logdir runs --host 0.0.0.0 --port 6006 > /tmp/tensorboard.log 2>&1 &
#   no navegador: http://localhost:3081/proxy/6006/   (pelo code-server; ou http://localhost:6006 no navegador do VNC)

# --- escolher a versão, exportar e avaliar ---
$PY eval_vec.py runs/sumo_ppo_v3/model.pt --vs runs/sumo_ppo_v1/model.pt runs/sumo_ppo_v2/model.pt   # rápido (GPU)
$PY tournament.py
$PY export_sumo.py runs/sumo_ppo_v3/model.pt exports/sumo/sumo_rl.py
$PY eval_sumo.py exports/sumo/sumo_rl.py --opponent exports/sumo/sumo_flanker.py --rounds 20 --side 0
$PY eval_sumo.py exports/sumo/sumo_rl.py --opponent exports/sumo/sumo_flanker.py --rounds 20 --side 1

# --- assistir, gravar, jogar ---
DISPLAY=:2 $PY eval_sumo.py exports/sumo/sumo_rl.py --opponent exports/sumo/sumo_chaser.py --viewer
./make_videos.sh
DISPLAY=:2 $PY teleop_sumo.py --fase 3

# --- conferências e gráficos ---
$PY check_parity.py
$PY plot_training.py && $PY plot_curriculum.py
```

**No psr-hero-kit, fora do container:** copie os arquivos `rl/exports/sumo/sumo_rl*.py`, `sumo_chaser.py` e
`sumo_flanker.py` para `psr-hero-kit/exercicios/sumo/` (eles precisam ter permissão de execução, `chmod +x`):

```bash
./run.sh run sumo robot0:=sumo_rl.py robot1:=none              # contra um robô parado
./run.sh run sumo robot0:=sumo_rl.py robot1:=sumo_flanker.py   # contra o flanker
./run.sh run sumo robot0:=sumo_rl.py robot1:=sumo_rl_v2.py     # fase 3 contra a fase 2
```

> **Atenção, GPU RTX 50xx:** a imagem `psr-kit-sim:gpu` do kit traz o `torch` cu126, que **não tem suporte à
> arquitetura Blackwell (sm_120)** da RTX 5070 Ti. O simulador do kit falha com
> `CUDA error: no kernel image is available`. Para usar o kit nesta máquina, rode a física na CPU:
> `./run.sh build cpu` uma vez e depois `./run.sh run sumo --cpu ...`. As políticas não mudam, e os testes da Seção 4.3
> foram feitos com o backend de CPU.

## 9. Arquivos

| Arquivo | Conteúdo |
|---|---|
| `rl/sumo_env.py` | ambiente vetorizado no Genesis (física, posição inicial, adversário, recompensa) |
| `rl/sumo_obs.py` | observação em torch e em numpy (a versão numpy é copiada para o nó) |
| `rl/opponents.py` | adversários com script (torch) e o grupo de cópias congeladas (`PolicyPool`) |
| `rl/ppo.py` | actor, critic, normalização e o PPO |
| `rl/train_sumo.py` | treino das fases 1, 2 e 3, registro em CSV e TensorBoard, checkpoints |
| `rl/export_sumo.py` | gera o nó ROS 1 de arquivo único com os pesos embutidos |
| `rl/eval_sumo.py` | coloca dois nós do kit para jogar no `psr_sim` real (sem janela ou com janela no VNC) |
| `rl/eval_vec.py`, `rl/tournament.py` | avaliação rápida na GPU e torneio para escolher a versão |
| `rl/record_video.py`, `rl/make_videos.sh` | gravação dos vídeos (câmera fora da tela) |
| `rl/teleop_sumo.py` | jogar com o teclado contra uma política |
| `rl/check_parity.py` | compara a física do ambiente vetorizado com a do `psr_sim` |
| `rl/plot_training.py`, `rl/plot_curriculum.py`, `rl/csv_to_tb.py` | gráficos e conversão de logs antigos para o TensorBoard |
| `rl/runs/` | checkpoints, configuração, log e TensorBoard de cada treino (`pool/` da fase 3) |
| `rl/results/` | resultados por rodada das avaliações no `psr_sim` |
| `rl/exports/sumo/sumo_rl.py` | **o nó final** (= fase 3), compatível com o kit |
| `rl/exports/sumo/sumo_rl_v1.py`, `_v2.py`, `_v3.py` | as três fases, para as comparações |
| `kit/exercicios/sumo/sumo_chaser.py`, `sumo_flanker.py` | adversários determinísticos de referência (o flanker não entra no treino) |
| `rl/docs/img/`, `rl/docs/videos/` | gráficos e vídeos |

## 10. Limitações e próximos passos

1. **Contra ela mesma, ainda é 50/50.** O que decide a rodada é qual lado consegue girar primeiro no contato. Mais
   iterações de *self-play*, ou um grupo maior de cópias, devem deixar a política mais robusta. Os empates por tempo
   praticamente sumiram.
2. **Vitórias demoradas contra quem se defende.** Contra o perseguidor e o flanker, cada vitória leva de 10 a 13 s
   em média (contra o robô parado, cerca de 4 s), porque a política gasta tempo contornando o adversário antes de
   empurrar. Ainda não foi testado um adversário que fique girando no centro só para se manter de frente. Contra ele, as
   rodadas podem chegar perto dos 40 s.
3. **Atraso e frequência.** O treino usa 20 Hz e sem atraso. No kit, a odometria chega a 25 Hz e há atraso de rede.
   Funcionou sem ajustes nos testes, mas treinar com um atraso aleatório de 0 a 2 passos (*domain randomization*) deixaria
   a política mais segura.
4. **Queda.** O treino encerra o episódio quando o robô cruza a borda, sem simular a queda. Isso não muda o resultado
   da rodada, porque o juiz usa a mesma regra.
5. **No kit há mais empates que no `psr_sim`.** A fase 3 foi testada no container ROS 1 do kit contra o
   `sumo_flanker.py` (cerca de 2 min de cada lado, física na CPU, em tempo real): como `hero_0`, 2 vitórias, 0 derrotas e
   2 empates; como `hero_1`, 4 vitórias, 0 derrotas e 0 empates. No total, **6 vitórias, 0 derrotas e 2 empates em 8
   rodadas**, enquanto no `psr_sim` sem atraso foram 39/0/1. A causa provável é o atraso de rede e a odometria a 25 Hz
   (item 3). O próximo passo para a competição é treinar com atraso aleatório.


---

# Parte 2: futebol (`soccer`)

## 11. Organização e currículo

O time tem três robôs controlados por **um único nó** (`rl/exports/soccer/soccer_rl.py`), com papéis fixos:

| Papel | Robô (azul / vermelho) | Controle |
|---|---|---|
| defensor / goleiro | `hero_0` / `hero_3` | goleiro com script (o defensor de RL foi treinado, Seção 14, mas não superou o script) |
| atacantes | `hero_1`, `hero_2` / `hero_4`, `hero_5` | **uma rede compartilhada** pelos dois (mesmos pesos, cada um com a sua observação) |

**Referencial do time:** cada time vê o campo como se atacasse para +x. Para o vermelho, as posições são giradas
em 180°. Assim, o mesmo arquivo joga dos dois lados (parâmetro `~team`), como exige o enunciado.

**Currículo**, igual ao do sumô:

| Estágio | Adversário (time vermelho) | Goleiro azul | Execução |
|---|---|---|---|
| A1 | parado | parado | `soccer_att_a1` (600 iterações, 39 milhões de passos, 7 min) |
| A2 | time com script (85%) + parado (15%), ganhos sorteados a cada episódio | script | `soccer_att_a2` (3 × 1000 iterações, com as correções da Seção 13) |
| A2w | igual ao A2 + potencial "bola longe das paredes" | script | `soccer_att_a2w` (1000 iterações, 14 min). **É a versão final** |
| A3 | *self-play*: cópias congeladas dos atacantes (65%) + scripts (35%) | script | `soccer_att_a3` (1500 iterações, 28 min) |
| D | atacantes vermelhos = **cópia congelada dos nossos atacantes (A3)**; o aprendiz é o `hero_0` | — | `soccer_def` (800 iterações, 20 min) |

## 12. Ambiente, observação e recompensa

- **Ambiente** (`rl/soccer_env.py`): 1024 campos em paralelo no Genesis, com 6 robôs, a bola de golfe (esfera plana com
  atrito de Coulomb aplicado como limite de força) e as **paredes do cenário**, criadas pelo próprio `World._add_static`
  do `psr_sim`. O episódio termina com um gol (regra do juiz) ou em 60 s. Para conferir a física, o mesmo time com script
  marca cerca de 10 gols em 2 min no ambiente de treino e 11 no `psr_sim`.
- **Observação** (`rl/soccer_obs.py`, 28 valores, no referencial do time; vetores relativos no referencial do robô):
  a própria pose e velocidade (6), a bola relativa ao robô, a posição absoluta da bola e a velocidade dela (6), os dois
  gols relativos ao robô (4), os dois companheiros (4), os três adversários ordenados por distância (6) e a ação anterior (2).
- **Actor e critic:** MLP 28 → 128 → 128 → 2 e 28 → 128 → 128 → 1 (tanh), separadas, com o mesmo PPO da Seção 3.
  O desvio-padrão da política passou a ficar limitado a [0,05; 1]. Sem esse limite ele cresceu até 2 no A1, porque a ação
  é cortada em ±1 e uma variância enorme não custa nada.
- **Recompensa dos atacantes** (a mesma para os dois):

| Termo | Valor |
|---|---|
| gol feito / sofrido | +10 / −10 |
| atacante parado (\|v\| < 0,02 m/s) | −0,02 por passo |
| bola travada (andou < 3 cm em 2 s) | −0,05 por passo (desde o A2) |
| *shaping*: `Φ = −3·d(bola, gol deles) − 1·min d(atacante, bola) [− 2·profundidade da bola na faixa de 0,15 m junto às paredes laterais (A2w)]` | potencial |

- **Recompensa do defensor:** gol sofrido −10; afastar a bola do nosso gol, em contato com ela, +0,5 por passo;
  sair da zona defensiva (x > −0,30) −0,05 por passo; bola travada no nosso campo −0,05 por passo; e o potencial
  `+2·d(bola, nosso gol) − 1·d(defensor, ponto de guarda)`, em que o ponto de guarda fica 12 cm à frente do gol, na
  linha gol–bola.

## 13. Problemas encontrados e correções

1. **A1 sem nenhum gol.** Nos primeiros 10 milhões de passos a política aprendeu só a andar reto, e o desvio-padrão
   caiu para 0,11 antes do primeiro gol. Empurrar uma bola redonda por 0,9 m até o gol, só explorando ao acaso, é raro
   demais. **Correção:** um currículo de **posições iniciais fáceis** (30% dos episódios começam com a bola a uma
   distância de 5 cm a 1,2 m do gol e um atacante atrás dela) e mais entropia (0,01). Com isso, o A1 chegou a 97% de
   episódios com gol e, no `psr_sim`, marcou **13 gols em 2 min** contra o time parado.
2. **Mínimo local no canto.** O potencial `−d(bola, gol)` dava quase o mesmo valor à bola presa no canto ao lado do
   gol e à bola perto dele. A política aprendeu a prender a bola ali. **Correção:** a distância passou a ser medida até
   um ponto dentro do gol, mais uma penalidade quando a bola sai da faixa do gol (|y| > 0,2).
3. **Travamentos nas paredes (problema principal, ainda não resolvido).** No ambiente vetorizado, a política do A2
   marca em 92% das saídas de bola contra o script com ganhos sorteados, e em 77% com os ganhos fixos do
   `soccer_script.py`. Ou seja, **em cerca de 22% das saídas de bola o jogo trava**: a bola fica presa contra uma parede
   lateral, com robôs dos dois times empurrando em sentidos opostos e com a mesma força. No treino isso só custa um
   episódio, que reinicia. Numa partida, nada reinicia o jogo (o kit não tem tempo de jogo nem faltas), e o primeiro
   travamento costuma durar até o fim. O arquivo exportado foi conferido: ele calcula a mesma observação e a mesma ação
   que o treino (diferença máxima de 10⁻⁶, `rl/check_soccer_obs.py`), então o problema não é de exportação.
   Tentativas:
   - penalidade de bola travada e episódios de 60 s: a fração de tempo travado não caiu (~35–45%);
   - posições iniciais com a bola encostada nas paredes e nos cantos;
   - **supervisor de regras** (`Unjam`, no `soccer_rl_hibrido.py`): se a bola anda < 3 cm em 3 s, os dois atacantes
     empurram juntos ao longo da parede por 2,5 s. Ele é acionado e a bola anda, mas continua colada na parede, porque
     nenhum robô consegue ficar entre ela e a parede. Não houve ganho mensurável;
   - potencial "bola longe das paredes" (A2w): foi a única mudança que melhorou o resultado nas partidas.

## 14. Resultados do futebol

**Partidas de 5 min no `psr_sim`** (`rl/eval_soccer.py`, os nós exportados com goleiro de script, contra
`soccer_script.py`, seeds 1, 2 e 3):

| Atacantes | Partida 1 | Partida 2 | Partida 3 | Total |
|---|---|---|---|---|
| A2 | 2 × 1 | 0 × 0 | 2 × 0 | 4 × 1 |
| A3 (*self-play*) | 0 × 0 | 0 × 0 | 0 × 0 | 0 × 0 |
| **A2w** (`soccer_rl.py`) | 0 × 0 | 1 × 0 | **9 × 1** | **10 × 1** |
| A2w + supervisor (`soccer_rl_hibrido.py`) | 0 × 0 | 1 × 0 | 8 × 2 | 9 × 2 |

Outros testes:
- A3 contra A2: 0 × 0 nas 3 partidas.
- `soccer_rl.py` jogando como **vermelho** contra o script: 0 × 0 (5 min).
- No **container ROS 1 do kit** (física na CPU), `blue:=soccer_rl.py red:=soccer_script.py`: roda sem erros, 0 × 0
  em 2,5 min.

**Defensor de RL × goleiro de script** (`rl/eval_defender_vec.py`, 512 episódios de 60 s, atacantes vermelhos = a
nossa rede do A3):

| Situação | Goleiro de script: gols sofridos | Defensor de RL: gols sofridos |
|---|---|---|
| saída de bola | 22,3% | 23,6% |
| chutes ao nosso gol (bola de 5 cm a 1,2 m do gol, atacante atrás) | 75,0% | 73,6% |

O defensor de RL **empatou com o script**. Por isso, o nó final usa o goleiro de script, e o exportador aceita a rede do
defensor com `--def runs/soccer_def/model.pt`.

### 14.1 Avaliação honesta

- O RL **aprendeu a fazer gols**: 13 gols em 2 min contra o time parado, e 9 × 1 numa partida sem travamento contra o
  script. Mas o resultado **depende de o jogo não travar nas paredes**, o que acontece na maioria das partidas.
- O *self-play* (A3) **piorou** o desempenho nas partidas: cópias da mesma política se travam entre si, e a política
  aprendeu a disputar a bola na parede, em vez de evitá-la.
- O defensor de RL não superou um goleiro simples por regra.
- **Para a competição:** o `soccer_rl.py` (A2w) é a melhor versão medida, mas com grande variação entre as partidas. Os
  próximos passos mais promissores são: (1) episódios contínuos de 5 min, sem reinício no treino, para os travamentos
  custarem o que custam numa partida; (2) um potencial mais forte contra a bola perto das paredes, também para o
  adversário (impedir que ele a leve para lá); (3) atribuir os papéis dinamicamente (quem está mais perto vai na bola,
  o outro apoia), como o `soccer_script.py` faz, em vez de dois atacantes iguais disputando a mesma bola.

### 14.2 Vídeos (`docs/videos/futebol_*.mp4`, gerados com `rl/make_soccer_videos.sh`)

São 90 s por vídeo no `psr_sim` real (câmera fora da tela), na seed 3, exceto o de travamento (seed 1). O azul é o
primeiro nome.

| Vídeo | Placar em 90 s |
|---|---|
| `futebol_a1_vs_parado.mp4` | 10 × 0 |
| `futebol_a2_vs_parado.mp4` | 8 × 0 |
| `futebol_final_vs_parado.mp4` | 8 × 0 |
| `futebol_a1_vs_script.mp4` | 0 × 0 |
| `futebol_a2_vs_script.mp4` | 2 × 1 |
| `futebol_final_vs_script.mp4` | 6 × 0 |
| `futebol_final_vs_script_travamento.mp4` | 1 × 0 (a bola trava na parede) |
| `futebol_a3_vs_script.mp4` | 0 × 0 |
| `futebol_hibrido_vs_script.mp4` | 5 × 0 |
| `futebol_final_vs_a2.mp4` | 0 × 0 |
| `futebol_a3_vs_final.mp4` | 4 × 2 |
| `futebol_script_vs_final.mp4` (final jogando de vermelho) | 0 × 0 |
| `futebol_script_vs_script.mp4` (referência) | 0 × 0 |

### 14.3 Crítico assimétrico (visão do campo, tempo de bola parada e Dijkstra)

O crítico só existe no treino, porque o `export_soccer.py` exporta apenas o ator. Por isso ele pode
receber informação que o ator não tem ("crítico privilegiado", ou assimétrico). Isso não fere as regras
do trabalho: o nó do kit continua com as mesmas 28 entradas. Implementação em `rl/soccer_critic.py`,
opção `--critic` do `train_soccer.py`, com normalização própria (`cnorm`) em `rl/ppo.py`.

| Variante | Entrada do crítico | Dim. |
|---|---|---|
| `sym` | as 28 entradas do ator | 28 |
| `asym` | 28 do ator + **visão do campo** (os 6 robôs: x, y, cos θ, sin θ, v, ω no referencial do time; bola: x, y, vx, vy; sem o ruído da observação) + **tempo de bola parada** (t/2 s, limitado a 3, e o indicador "travada") | 70 |
| `asym_dij` | o anterior + 5 distâncias calculadas por **Dijkstra**: eu → ponto atrás da bola, companheiro → ponto atrás da bola, desvio em relação à linha reta, bola → gol contornando os adversários, desvio da bola | 75 |

A "visão do campo" é o vetor com todas as posições, e não uma imagem com rede convolucional. O simulador já
entrega as coordenadas exatas, então uma imagem não teria informação a mais.

**Dijkstra.** A grade tem células de 4 cm (45 × 30) e 16 vizinhos (8 + os saltos de cavalo). O custo de
uma aresta é o comprimento vezes o custo médio das duas células: 1 em célula livre, 15 dentro de um
obstáculo. O obstáculo é caro mas não proibido, porque um robô pode ser empurrado. Os obstáculos são, para o
robô, a bola (que precisa ser contornada para chegar atrás dela), o goleiro e os 3 adversários; para a bola,
os 3 adversários. Para rodar em lote nos 1024 ambientes, a fila de prioridade é trocada por relaxações
paralelas de todas as células (Bellman-Ford na grade). Depois de 40 varreduras o resultado é o mesmo do
Dijkstra: a diferença para 200 varreduras foi 0. O erro é de 0,3 % numa reta de 1 m e de 3 % ao contornar
um disco (1,079 m contra 1,045 m do contorno ideal). Cada campo leva 1,6 ms para 1024 ambientes
(`torch.compile`), mas o passo do ambiente sobe de 14 para 24 ms.

**Experimento.** Mesmas condições do A2w (estágio A2, `--wall 2`, a partir do `soccer_att_a2`, 1000
iterações, seed 0). Os críticos assimétricos começam do zero, então o controle justo é o `sym` também
com crítico novo (`--fresh-critic`). O A2w original herdou o crítico do A2. Para reproduzir:
`rl/eval_critics.sh` e `rl/eval_critics_sim.sh`.

| | A2w original (sym, crítico herdado) | sym, crítico novo | asym | asym + Dijkstra |
|---|---|---|---|---|
| Variância explicada do crítico, it. 1–50 | – | 0,76 | **0,85** | 0,83 |
| Variância explicada, it. 901–1000 | – | 0,933 | 0,946 | **0,951** |
| Perda do crítico, it. 901–1000 | 0,133 | 0,147 | 0,127 | **0,119** |
| Vetorizado (512 saídas de bola, script com ganhos aleatórios): gol / sofrido | **92,6 % / 0,4 %** | 81,2 % / 3,9 % | 79,7 % / 1,0 % | 80,1 % / 0,8 % |
| Vetorizado, script fixo (como no kit): gol / sofrido | **70,7 % / 6,4 %** | 59,4 % / 14,1 % | 50,0 % / 7,2 % | 57,8 % / 3,9 % |
| psr_sim, 6 × 5 min contra `soccer_script` (seeds 1–6) | **16 × 1** | 12 × 1 | 4 × 1 | 4 × 0 |

Placares por seed no psr_sim:
- A2w: 1-0, 0-0, 9-1, 2-0, 4-0, 0-0;
- sym novo: 0-0, 0-0, 3-1, 1-0, 0-0, 8-0;
- asym: 0-0, 2-0, 1-0, 0-0, 0-0, 1-1;
- Dijkstra: 0-0, 0-0, 4-0, 0-0, 0-0, 0-0.

**Análise.**
1. **O crítico fica melhor.** Com a visão do campo e o tempo de bola parada, o crítico prevê o retorno
   melhor e mais cedo (0,85 contra 0,76 de variância explicada nas primeiras 50 iterações). O Dijkstra
   acrescenta pouco: 0,951 contra 0,946, e perda 0,119 contra 0,127. Faz sentido, porque com só 4 obstáculos
   pequenos a distância geodésica quase sempre é a reta. O desvio médio foi de 0,02 m para o robô e
   0,06 m para a bola. Além disso, uma MLP com todas as posições já consegue aproximar boa parte disso.
2. **Isso não virou um ator melhor.** O crítico não muda a política ótima: só torna a estimativa da
   vantagem menos ruidosa. Nas 1000 iterações, nenhuma variante com crítico novo alcançou o A2w. Trocar o
   crítico tem um custo no início: com o crítico recém-iniciado, as vantagens ficam ruins e o ator perde
   parte do que tinha aprendido no A2. Entre os de crítico novo, os assimétricos sofrem menos gols
   (vetorizado com script fixo: 3,9 % com Dijkstra contra 14,1 %), mas também marcam menos. Ficaram mais
   conservadores.
3. **O travamento na parede não mudou.** A fração de passos com bola parada ficou em 37–40 % nas três
   variantes. Saber há quanto tempo a bola está parada ajuda o crítico a prever um retorno baixo nessas
   situações, mas não ensina o ator a sair delas, porque o ator continua sem essa informação.
4. **Vale a pena o Dijkstra no crítico?** Por enquanto não: +70 % de tempo por passo para +0,005 de
   variância explicada, sem ganho no ator. Onde ele pode ajudar mais:
   - (a) no **potencial da recompensa**, trocando a distância euclidiana robô → bola por "robô → ponto atrás
     da bola contornando a bola". Isso afeta o ator diretamente e penaliza chegar à bola pelo lado errado,
     que é uma das causas do travamento;
   - (b) como **entrada do ator**. Aí o cálculo teria de ir para o nó; em numpy com `heapq`, numa grade
     de 1350 células, cabe em 20 Hz.

Conclusão: o nó final continua sendo o A2w (`soccer_rl.py`). O crítico assimétrico fica disponível
(`--critic asym|asym_dij`). Ele deve servir melhor em treinos longos que já partam dele, por exemplo no
self-play A3, em que um crítico que vê o adversário inteiro tem mais o que aproveitar.

## 15. Como rodar (futebol)

```bash
cd ~/ros2_ws/rl; PY=~/ros2_ws/.venv/bin/python; S=exports/soccer
# treino
$PY train_soccer.py --stage A1 --iters 600  --name soccer_att_a1
$PY train_soccer.py --stage A2 --iters 1000 --name soccer_att_a2  --init runs/soccer_att_a1/model.pt
$PY train_soccer.py --stage A2 --iters 1000 --name soccer_att_a2w --init runs/soccer_att_a2/model.pt --wall 2.0
$PY train_soccer.py --stage A3 --iters 1500 --name soccer_att_a3  --init runs/soccer_att_a2/model.pt
$PY train_soccer.py --stage A2 --iters 1000 --name soccer_att_a2w_dij --init runs/soccer_att_a2/model.pt --wall 2.0 --critic asym_dij
$PY train_soccer.py --stage D  --iters 800  --name soccer_def     --attacker runs/soccer_att_a3/model.pt
# exportar (um arquivo, numpy): --def para usar o defensor de RL; --no-unjam para a rede pura
$PY export_soccer.py --att runs/soccer_att_a2w/model.pt --no-unjam --out $S/soccer_rl.py
# avaliar e gravar
$PY eval_soccer_vec.py runs/soccer_att_a2w/model.pt
$PY eval_soccer.py --blue $S/soccer_rl.py --red $S/soccer_script.py --minutes 5 --matches 3   # 1 subprocesso por partida
$PY record_soccer.py --blue $S/soccer_rl.py --red $S/soccer_script.py --seconds 90 --title "..." --out docs/videos/futebol.mp4
DISPLAY=:2 $PY eval_soccer.py --blue $S/soccer_rl.py --red $S/soccer_script.py --viewer      # assistir no VNC
```

No kit: copie `soccer_rl.py` e `soccer_script.py` para `psr-hero-kit/exercicios/soccer/` e rode
`./run.sh run soccer blue:=soccer_rl.py red:=soccer_script.py --cpu`.

| Arquivo | Conteúdo |
|---|---|
| `rl/soccer_env.py`, `rl/soccer_obs.py`, `rl/soccer_scripts.py` | ambiente, observação (torch e numpy) e time com script em torch |
| `rl/train_soccer.py`, `rl/frozen_pool.py` | treino (A1, A2, A3, D) e as cópias congeladas |
| `rl/export_soccer.py` | gera o nó de arquivo único (atacantes + goleiro ou defensor de RL + supervisor opcional) |
| `rl/eval_soccer.py`, `rl/eval_soccer_vec.py`, `rl/eval_defender_vec.py`, `rl/check_soccer_obs.py` | avaliações e a conferência do arquivo exportado |
| `rl/record_soccer.py` | vídeo de uma partida |
| `rl/soccer_critic.py` | entrada do crítico assimétrico (visão do campo, bola parada, Dijkstra em grade na GPU) |
| `rl/eval_critics.sh`, `rl/eval_critics_sim.sh` | comparação dos críticos (seção 14.3) |
| `rl/exports/soccer/soccer_rl_critic_*.py` | nós exportados das variantes do crítico (só para a comparação) |
| `rl/exports/soccer/soccer_rl.py` | **nó final** (atacantes A2w + goleiro de script) |
| `rl/exports/soccer/soccer_rl_hibrido.py` | o mesmo, com o supervisor de travamento |
| `rl/exports/soccer/soccer_rl_a1.py`, `_a2.py`, `_a3.py` | versões anteriores, para as comparações |
| `kit/exercicios/soccer/soccer_script.py` | time determinístico de referência |

## 16. Futebol depois do torneio: bola presa (V1) e sensores com ruído + fusão (V2)

Duas versões novas, as duas partindo da versão final (A2w):

| Versão | Execução | Nó exportado |
|---|---|---|
| **V1: teleporte da bola** | `soccer_att_a2w_tp` (A2w + regra da bola presa, 1000 it., 46 min) | `exports/soccer/soccer_rl_tp.py` |
| **V2: teleporte + ruído + fusão** | `soccer_att_a2w_tp_fus` (V1 + sensores ruidosos + fusão odometria × lidar, crítico `truth`, 1500 it., 38 min) | `exports/soccer/soccer_rl_tp_fusao.py` |

### 16.1 A regra da bola presa do servidor do torneio (`rl/soccer_jam.py`)

O `psr_sim` do kit não tem essa regra. Ela foi deduzida dos 646 eventos "Bola presa: reposta no ponto neutro"
gravados ao vivo (`torneio_logs/dados/ao_vivo/soccer`):

- **disparo:** a bola fica ~10 s quase no mesmo lugar, quase sempre (≈95 %) prensada contra a parede lateral por
  um robô. No treino: âncora = posição da bola; se a bola sai de um raio `r` da âncora, a âncora muda e o tempo
  recomeça; após `T` s, teleporte. `T` ~ U(9, 11) s e `r` ~ U(4, 7) cm por episódio, porque o critério exato do
  servidor não dá para recuperar com amostras de ~3 Hz (o melhor ajuste reproduz ~70 % dos eventos);
- **destino:** o ponto neutro mais próximo da bola, entre (±0,45, ±0,30) e (0, 0), que não tenha robô a menos de
  0,15 m (acerta 606 dos 646 eventos);
- **depois:** a bola fica **parada no ponto neutro por 2 s**, e os robôs continuam jogando (não são congelados nem
  reposicionados).

`--jam` no `train_soccer.py` liga a regra no ambiente vetorizado; `eval_soccer.py --jam` a aplica no `psr_sim` real.

### 16.2 Ruído e fusão (`rl/soccer_fusion.py`)

O professor ainda não divulgou quanto ruído o kit vai ter. Por isso o **nível** `s` é sorteado por episódio
(s = 0 em 20 % dos episódios, senão U(0, 2)). Com s = 1 (nominal):

| Sensor | Modelo |
|---|---|
| odometria (pose) | pose real + **deriva** (passeio aleatório: 1 cm/√s em x, y; 0,02 rad/√s em yaw) + ruído branco (1 cm; 0,03 rad) |
| odometria (v, ω) | real + ruído branco (0,01 m/s; 0,1 rad/s) |
| lidar (8 feixes, 0,5 m) | mesmo modelo do `psr_sim` (paredes, os outros robôs e a bola) + ruído de 1 cm; 2 % dos feixes perdidos (= alcance máximo) |
| bola | posição + 0,5 cm, velocidade + 0,03 m/s |

**Fusão**, igual no treino (torch, em lote) e no nó (numpy; diferença máxima de 2·10⁻⁶ em `rl/check_fusion.py`):

- **nossos 3 robôs: EKF** com estado (x, y, θ, bₓ, b_y, b_θ), ou seja, a pose e a **deriva da odometria**.
  Predição com (v, ω) da odometria; correção 1: a pose da odometria (z = pose + deriva); correção 2: **os 8 feixes do
  lidar contra as paredes conhecidas** (traçado de raios no mapa do campo, com o jacobiano analítico
  ∂r/∂p = −n/(n·u), ∂r/∂θ = −r (n·u′)/(n·u)). Feixes que batem num robô ou na bola ficam mais curtos que a parede e
  são rejeitados por um teste χ² (3σ); feixes sem parede ao alcance não são usados. Perto das paredes o lidar torna
  a deriva observável, e o EKF a corrige; no meio do campo ele segue a odometria. Um salto > 0,25 m (saída de bola
  depois de um gol) reinicia a pose e mantém a deriva;
- **bola:** filtro de Kalman de velocidade constante, reiniciado em saltos > 10 cm (teleporte, saída de bola);
- **adversários:** suavização exponencial da posição (só temos a odometria deles);
- **nível de ruído estimado em tempo real:** o resíduo de yaw da odometria, θₖ − θₖ₋₁ − ω·dt, não é afetado pelos
  empurrões (ω é medido), então a média exponencial de e² dá o nível s. O filtro escala Q e R por max(s, 0,3).
  No nó, `~noise_level:=-1` (padrão) estima o nível; `~noise_level:=<s>` o fixa, quando o professor divulgar os valores.

Erro das estimativas (`check_fusion.py`, time de script, 256 campos):

| Nível | estimado | pose: odometria → EKF | yaw: odometria → EKF | bola: medida → KF |
|---|---|---|---|---|
| 0 | 0,37 | 0 → 0,73 cm | 0 → 1,3° | 0 → 0,04 cm |
| 0,5 | 0,59 | 2,7 → 1,2 cm | 2,3 → 2,1° | 0,35 → 0,18 cm |
| 1 | 1,02 | 5,7 → 1,9 cm | 4,7 → 3,5° | 0,71 → 0,35 cm |
| 2 | 1,92 | 10,7 → 2,6 cm | 9,5 → 5,1° | 1,41 → 0,69 cm |

Sem a estimativa do nível, um filtro calibrado para s = 1 errava 1,3 cm e 2,9° com ruído zero, pior que a
odometria crua. Isso motivou o filtro adaptativo.

**Treino da V2:** a política vê só o estado estimado; o crítico (`--critic truth`) vê as mesmas 28 entradas
calculadas com o estado verdadeiro, herdando os pesos do crítico da V1. O goleiro de script também age sobre o
estado estimado, como no nó.

### 16.3 Resultados

**Vetorizado** (`rl/eval_soccer_match_vec.py`, 512 partidas de 3 min contra o time de script, regra da bola presa,
saldo médio de gols por partida ± erro padrão; `results/soccer_tp_fus/match_vec.jsonl`):

| Modelo | Sensores | Nível 0 | 0,5 | 1 | 2 |
|---|---|---|---|---|---|
| A2w (final anterior) | verdade, ganhos fixos do kit | +4,29 ± 0,15 | | | |
| **V1** | verdade, ganhos fixos do kit | **+4,67 ± 0,15** | | | |
| A2w | verdade, ganhos aleatórios | +9,62 | | | |
| **V1** | verdade, ganhos aleatórios | **+9,94** | | | |
| A2w | ruído **sem filtro** | +4,01 | +2,12 | −0,71 | −4,48 |
| A2w | ruído + fusão | +3,49 | +2,82 | +1,56 | −0,36 |
| V1 | ruído + fusão | +4,17 | +4,03 | +2,97 | +0,64 |
| **V2** | ruído + fusão | **+4,59** | **+4,21** | **+3,16** | **+1,31** |

- A regra da bola presa muda o jogo: com ela, a A2w faz 4,84 × 0,54 por partida, contra 3,33 × 0,30 sem ela. A V1
  aprendeu a tirar proveito dela: menos gols sofridos (0,30), menos tempo de bola travada (17 % contra 20 %) e menos
  teleportes (1,6 contra 1,9 por partida).
- Sem filtro, o ruído derruba a política: no nível 1 ela já perde (58,6 % de derrotas). Só a fusão, sem retreinar,
  recupera boa parte; treinar com ela (V2) é o melhor em todos os níveis e **não custa nada com ruído zero**
  (+4,59 contra +4,67 da V1 com a verdade).
- O `best.pt` das duas execuções é de iterações iniciais (critério do treino, com saídas fáceis); use o `model.pt`.

**`psr_sim` real** (`eval_soccer.py --jam`, contra `soccer_script.py`, partidas de 3 min):

| Nó (azul) | Condição | Partidas | Gols |
|---|---|---|---|
| `soccer_rl.py` (A2w) | bola presa | 6 | 21 × 3 |
| `soccer_rl_tp.py` (V1) | bola presa | 6 | **32 × 1** |
| `soccer_rl_tp.py` (V1) **de vermelho** | bola presa | 3 | 4 × 0 |
| `soccer_rl_tp.py` (V1, sem filtro) | bola presa + ruído nível 1 | 4 | 8 × 3 |
| `soccer_rl_tp_fusao_v1.py` (V1 + fusão) | bola presa + ruído nível 1 | 4 | 26 × 0 |
| `soccer_rl_tp_fusao.py` (V2) | bola presa + ruído nível 1 | 4 | 23 × 0 |

No `psr_sim` o adversário também recebe os dados ruidosos e não tem filtro, por isso os placares são altos. Com 4
partidas, a diferença entre V1 + fusão e V2 está dentro da variação; a comparação vetorizada (512 partidas) mostra a
vantagem da V2, maior quanto mais ruído.

**Qual usar:** com o kit atual (sem ruído), `soccer_rl_tp.py` (V1) ou `soccer_rl_tp_fusao.py` (V2), que são
equivalentes. Quando o kit tiver ruído, `soccer_rl_tp_fusao.py` (V2), de preferência com `~noise_level` fixado nos
valores divulgados (convertidos para o nível s da tabela da seção 16.2), ou no modo automático.

**Limitações:** o modelo de ruído é uma aposta (deriva + ruído branco em todos os sensores). Se o ruído real tiver
outra forma, por exemplo só no lidar ou com atraso, o filtro continua válido, mas a estimativa do nível pelo yaw pode
errar, e convém fixar `~noise_level`. A regra da bola presa foi deduzida das gravações, não do código do servidor.

```bash
cd ~/ros2_ws/rl; PY=~/ros2_ws/.venv/bin/python; S=exports/soccer
$PY check_fusion.py                                                     # paredes, paridade numpy x torch, erros, teleporte
$PY train_soccer.py --stage A2 --iters 1000 --wall 2 --jam --init-std 0.25 --name soccer_att_a2w_tp --init runs/soccer_att_a2w/model.pt
$PY train_soccer.py --stage A2 --iters 1500 --wall 2 --jam --fusion --critic truth --init-std 0.25 \
    --name soccer_att_a2w_tp_fus --init runs/soccer_att_a2w_tp/model.pt
$PY export_soccer.py --att runs/soccer_att_a2w_tp/model.pt --no-unjam --out $S/soccer_rl_tp.py
$PY export_soccer.py --att runs/soccer_att_a2w_tp_fus/model.pt --no-unjam --fusion --out $S/soccer_rl_tp_fusao.py
$PY eval_soccer_match_vec.py runs/soccer_att_a2w_tp/model.pt runs/soccer_att_a2w_tp_fus/model.pt --jam --sensing fusion --levels 0 0.5 1 2
$PY eval_soccer.py --blue $S/soccer_rl_tp_fusao.py --red $S/soccer_script.py --minutes 3 --matches 4 --jam --noise 1
```

| Arquivo novo | Conteúdo |
|---|---|
| `rl/soccer_jam.py` | regra da bola presa (torch para o treino, numpy para o `psr_sim`) |
| `rl/soccer_fusion.py` | modelo de ruído (torch e numpy) e os filtros: EKF odometria × lidar, KF da bola, suavização, estimador do nível |
| `rl/check_fusion.py` | conferências: paredes, paridade numpy × torch, erros das estimativas, teleporte |
| `rl/eval_soccer_match_vec.py` | partidas de 3 min vetorizadas, com regra da bola presa e níveis de ruído |
| `results/soccer_tp_fus/` | resultados (JSONL vetorizado e CSVs do `psr_sim`) |

## 17. Sumô com sensores ruidosos + fusão (v6)

A mesma técnica da seção 16.2, adaptada ao sumô. A v6 parte da v5 (`runs/sumo_ppo_v5/model.pt`): mesma fase 5,
mesmo grupo de cópias congeladas, mesma rede (entradas 12 → 64 → 64 → 2). Muda só o que a política enxerga, que
agora é o estado **estimado** a partir de sensores com ruído.

| Versão | Execução | Nó exportado |
|---|---|---|
| **v6: ruído + fusão** | `sumo_ppo_v6_fus`, iteração 800 de 1500 (36 min; `pool/it0800.pt`) | `exports/sumo/sumo_rl_v6_fusao.py` |
| v5 + fusão, sem retreino (comparação) | `sumo_ppo_v5/model.pt` | `exports/sumo/sumo_rl_v5_fusao.py` |

### 17.1 Ruído e fusão (`rl/sumo_fusion.py`)

**Ruído (`SumoSensorNoise`).** Os valores nominais são os do futebol (nível s = 1), e o nível é sorteado por
episódio: s = 0 em 20 % dos episódios, nos demais U(0, 2).

- **Odometria (pose):** a pose verdadeira mais uma deriva (passeio aleatório de 1 cm/√s; 0,02 rad/√s) e ruído
  branco (1 cm; 0,03 rad).
  - A deriva **não começa em zero**, porque uma rodada é uma entre várias de uma partida. A deriva inicial é a de
    U(0, 120) s de passeio, ou U(0, 40) s nos inícios aleatórios e em contato, que representam momentos do meio
    de uma rodada.
- **Odometria (v, ω):** os valores verdadeiros mais ruído (0,01 m/s; 0,1 rad/s).
- **Lidar (8 feixes):**
  - no sumô os feixes só batem no adversário, porque o ringue não tem paredes;
  - os feixes que acertam algo recebem 1 cm de ruído; um feixe que não acerta nada lê o alcance máximo;
  - 2 % dos feixes se perdem.

**Fusão (`SumoEKF` em numpy no nó, `SumoFusionTorch` em lote no treino).** A diferença máxima entre as duas
versões é de 2·10⁻⁶ (`check_sumo_fusion.py`).

É um único EKF com os dois robôs. O estado é (x, y, θ, bₓ, b_y, b_θ) × 2, isto é, a pose e a deriva da odometria
de cada robô.

- **Predição:** cada robô avança com o (v, ω) da sua odometria.
- **Correção 1:** as poses de odometria dos dois robôs, com z = pose + deriva.
- **Correção 2, lidar:** cada feixe que acertou alguma coisa acertou o adversário.
  - O centro do adversário fica em ≈ p + (r + R_robô)·d. A medida é precisa ao longo do feixe e tem incerteza de
    ±R_robô/√3 na direção perpendicular, porque o feixe pode acertar fora do centro.
  - Isso dá a posição do adversário **em relação a nós**, sem a deriva de nenhuma das duas odometrias.
  - O lidar não corrige o nosso θ: o erro de θ só aumenta a variância perpendicular.
- **Correção 3, saída da rodada:** no início de cada rodada os dois robôs estão em posições conhecidas
  (x = ∓0,18; y ≈ 0 ± 0,035; frente a frente). Isso torna observável a deriva da pose absoluta, ou seja, a
  distância até a borda.
  - O início da rodada é detectado por um salto da odometria (o teleporte) ou avisado pelo `/game/state`.
  - A correção é aplicada quando a pose estimada fica a menos de 15 cm do ponto de saída, e o filtro tenta por
    até 1 s.
- **Nível de ruído:** estimado em tempo real pelo resíduo de θ, como no futebol. No nó, o parâmetro
  `~noise_level:=-1` (padrão) faz o nó estimar o nível; um valor ≥ 0 fixa o nível.

**Duas tentativas descartadas:**

- **Restrição exata "o ponto atingido está no círculo do adversário", com θ no jacobiano.** Com derivas
  grandes, o filtro explicava o erro girando θ: o erro de θ foi para 60–75°.
- **Ruído aplicado também aos feixes sem retorno.**
  - Com s = 1, cerca de 30 % desses feixes liam ~0,49 m e pareciam acertos falsos.
  - Esse defeito afetou um primeiro treino, que foi interrompido e está guardado em
    `runs/_sumo_ppo_v6_fus_bug_lidar`.
  - Além de deixar esses feixes sem ruído, o filtro passou a só aceitar como acerto um feixe abaixo de
    RMAX − 3σ_r.

**Erro médio das estimativas** (`check_sumo_fusion.py`, `SumoEnv` com adversários de script):

| Nível | Nível estimado | Nossa posição: odometria → EKF | Adversário em relação a nós | Distância ao centro |
|---|---|---|---|---|
| 0 | 0,38 | 0 → 0,47 cm | 0 → 0,43 cm | 0 → 0,27 cm |
| 0,5 | 0,57 | 4,8 → 2,4 cm | 6,7 → 0,9 cm | 3,0 → 1,3 cm |
| 1 | 1,00 | 9,2 → 5,5 cm | 12,9 → 2,1 cm | 6,0 → 3,4 cm |
| 2 | 1,87 | 18,6 → 14,3 cm | 25,8 → 11,7 cm | 12,7 → 9,3 cm |

**Treino:**

- A política vê o estado estimado. O crítico (assimétrico) vê as mesmas 12 entradas calculadas com o estado
  verdadeiro; o crítico e a normalização vieram da v5.
- Os adversários (scripts e cópias congeladas) jogam com o estado verdadeiro.

### 17.2 Resultados

**Ambiente vetorizado** (`eval_vec.py --summary`): 512 rodadas por adversário, política determinística, saída
padrão. Os 9 adversários são os 7 scripts, a v5 e a v3. Cada célula é vitória / derrota em %; os arquivos estão em
`results/sumo_fus/vec_*.txt`.

| Modelo | Sensores | Verdade | Nível 0 | 0,5 | 1 | 2 |
|---|---|---|---|---|---|---|
| v5 | odometria ruidosa, **sem filtro** | 78 / 1 | 78 / 1 | 27 / 59 | 9 / 82 | 2 / 95 |
| v5 | ruído + fusão | | 73 / 1 | 74 / 3 | 72 / 10 | 49 / 39 |
| **v6 (it. 800)** | ruído + fusão | 71 / 0 | 72 / 0,1 | 73 / 0 | **72 / 2,6** | **52 / 30** |
| v6 it. 1000 / 1200 / 1500 | ruído + fusão | | 69 / 66 / 69 | | 68 / 68 / 67 | 50 / 49 / 49 |

- **Sem filtro, o ruído destrói a v5.** Com a borda estimada errada, o robô sai sozinho do ringue: no nível 1 são
  82 % de derrotas.
- **O filtro sozinho**, sem retreino, já recupera quase tudo até o nível 1.
- **Treinar com o filtro (v6)** corta as derrotas: no nível 1 elas caem de 10 % para 2,6 %, e no nível 2 de 39 %
  para 30 %.
- **Escolha da iteração 800:** depois dela o desvio padrão da ação angular satura em 1 e aumentam os empates. Por
  isso foi exportada a iteração 800, não o `model.pt`.

**`psr_sim` real** (`eval_sumo.py --noise`): 10 rodadas por confronto. Os dois nós recebem os sensores ruidosos
(`SumoSensorNoiseNp`, deriva acumulada ao longo das rodadas). O resultado é do primeiro nó (vitórias, derrotas,
empates); os arquivos estão em `results/sumo_fus/*.csv` e `psr_sim.txt`.

| Confronto (hero_0 × hero_1) | Sem ruído | Ruído 1 | Ruído 2 |
|---|---|---|---|
| v6 × parado | 10–0–0 | 10–0–0 | 9–1–0 |
| v5 × parado | 10–0–0 | 2–1–7 | |
| v5 + fusão × parado | | 10–0–0 | |
| v6 × v5 | 10–0–0 | 10–0–0 | 10–0–0 |
| v5 × v6 | 0–9–1 | 0–10–0 | |
| v6 × v5 + fusão | 10–0–0 | 9–0–1 | |

- **v6 contra a v5:** a v6 vence a v5 dos dois lados. Com ruído 1 e a v6 como hero_1, foram 10 vitórias em 10.
  A v6 vence até **sem ruído**, ou seja, o treino com ruído também melhorou o jogo dela.
- **Ressalva:** são só 10 rodadas por confronto, contra uma única política.

**Qual usar:** `sumo_rl_v6_fusao.py`, que precisa do `/hero_<robot>/scan` (o kit já publica esse tópico).

**Limitações:**

- **Modelo de ruído:** é uma aposta, como no futebol.
- **Posição absoluta:** só vem da odometria e da saída de cada rodada, porque o lidar não vê a borda. Com ruído
  alto (s = 2), a distância à borda ainda erra ~9 cm.
- **Saída da rodada:** o filtro supõe a saída padrão do `psr_sim` (∓0,18; frente a frente). Se o servidor mudar
  isso, convém desligar a correção 3.

```bash
cd ~/ros2_ws/rl; PY=~/ros2_ws/.venv/bin/python; S=exports/sumo
$PY check_sumo_fusion.py                                         # paridade numpy x torch, erros por nível
$PY train_sumo.py --phase 5 --fusion --iters 1500 --name sumo_ppo_v6_fus --init runs/sumo_ppo_v5/model.pt \
    --pool-from runs/sumo_ppo_v5/pool
$PY export_sumo.py runs/sumo_ppo_v6_fus/pool/it0800.pt $S/sumo_rl_v6_fusao.py --fusion
$PY eval_vec.py runs/sumo_ppo_v6_fus/pool/it0800.pt --vs runs/sumo_ppo_v5/model.pt runs/sumo_ppo_v3/model.pt \
    --sensing fusion --levels 0 0.5 1 2 --summary
$PY eval_sumo.py $S/sumo_rl_v6_fusao.py --opponent $S/sumo_rl_v5.py --noise 1 --rounds 10
./avaliar_sumo_fus.sh                                            # tudo acima
```

| Arquivo | Conteúdo |
|---|---|
| `rl/sumo_fusion.py` (novo) | modelo de ruído (torch e numpy), EKF dos dois robôs e estimador do nível |
| `rl/check_sumo_fusion.py` (novo) | paridade numpy × torch (com teleporte) e erros das estimativas |
| `rl/avaliar_sumo_fus.sh` (novo) | export e avaliações da v6 |
| `rl/sumo_env.py` | `sensing = truth / raw / fusion`, `critic_obs()` |
| `rl/train_sumo.py` | `--fusion` (crítico com o estado verdadeiro), `--noise-max` |
| `rl/export_sumo.py` | `--fusion`: o nó assina o `/scan`, roda o EKF, parâmetro `~noise_level` |
| `rl/eval_sumo.py` | `--noise` (ruído no `psr_sim` real), feixes do lidar e aviso de nova rodada para o nó |
| `rl/eval_vec.py` | `--sensing`, `--levels`, `--summary` |
