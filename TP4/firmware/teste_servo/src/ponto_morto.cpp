// TP4 - Controle de velocidade da roda (Arduino Mega 2560).
//
// Encoder (X4) nos pinos 2 (A) e 3 (B). Servo SG90 de giro continuo no pino 10
// (header Servo1 do shield de motores; confirmado no teste de sanidade).
//
// Malha: v = k*dticks/dt (dt medido com micros()) -> filtro -> PID -> correcao
// (us) somada ao neutro. Correcao limitada a +-250 us, pulso entre 1000 e 2000.
//
// Convencao: pulso > neutro deve produzir ticks crescentes (v > 0). Se sair
// invertido, troque A e B no encoder (Encoder enc(3, 2)) e registre no relatorio.
//
// Comandos seriais (115200, terminar com \n). Um comando por linha:
//   h              ajuda            ?              mostra parametros
//   n <us>         neutro medido    o <us>         malha aberta, pulso fixo
//   s <mm/s>       malha fechada, setpoint fixo (muda o setpoint zera a integral)
//   x              para (pulso = neutro)           z   zera ticks e integral
//   g kp ki kd     ganhos [us/(mm/s), us/(mm/s)/s, us/(mm/s^2)]
//   f 0|1          filtro no PID: 0 = media movel, 1 = passa-baixa
//   w <n>          janela da media movel (1..10)   a <alfa>  alfa do passa-baixa
//   q v1..v5       patamares do ensaio 1 (mm/s)    r <mm/s>  velocidade dos ensaios 2 e 3
//   u <pino>       troca o pino do sinal do servo (9 ou 10), para teste
//   l 0|1          log tambem em repouso
//   N [ini] [fim]  BUSCA DO NEUTRO: varre de ini a fim (default 1400..1600 us) em
//                  passos de 10 us, 5 s cada, e imprime a tabela e o neutro
//   A [passo] [s]  curva em malha aberta: 1000..2000 us, passo (default 50 us),
//                  cada pulso por s segundos (default 3)
//   1              ensaio 1: cinco patamares       2   ensaio 2: inversao de sentido
//   3 [voltas]     ensaio 3: distancia (default 3 voltas a velocidade r)
//
// Log CSV: t_s,ticks,s_mm,v_bruta,v_filtrada,setpoint,pulso_us,dt_us,v_mm,v_lp
// (v_filtrada = a do filtro escolhido; v_mm e v_lp sao as duas saidas, para a
// comparacao de filtros). Linhas que comecam com '#' sao comentarios.

#include <Arduino.h>
#include <Encoder.h>
#include <Servo.h>

// ---------------- constantes medidas na bancada ----------------
#define D_MM        43.40f              // paquimetro
#define N_TICKS     287.3f              // media de 20 voltas a mao, X4
static const float K_MM_TICK = PI * D_MM / N_TICKS;   // mm/tick

static float neutro_us = 1500.0f;       // PREENCHER apos a busca do neutro (comando N)

// ---------------- configuracao ----------------
static uint8_t pin_servo = 10;          // 'u <pino>' troca em tempo de execucao
static const unsigned long TS_US = 50000UL;      // periodo nominal da amostra
static const float LIM_CORR_US = 250.0f;
static const float PULSO_MIN = 1000.0f, PULSO_MAX = 2000.0f;

static const float PAT_S   = 3.0f;      // duracao de cada patamar (ensaio 1)
static const float INV_S   = 4.0f;      // duracao de cada sentido (ensaio 2)
static const float PARADA_S = 1.5f;     // log apos parar (ensaio 3)
static const float TMAX_E3 = 40.0f;

static const float NEU_S = 5.0f;        // duracao de cada pulso na busca do neutro
static const float NEU_DESC_S = 1.0f;   // transitorio descartado em cada pulso
static const long  NEU_TICKS_PARADA = 2;// |ticks| ate isso em ~4 s = roda parada

static float pat[5] = {30, 50, 70, 90, 110};   // mm/s, AJUSTAR com q
static float v_ens = 60.0f;                    // mm/s, ensaios 2 e 3 (r)

static float kp = 0.0f, ki = 0.0f, kd = 0.0f;  // sintonizar na bancada (g)

// ---------------- estado ----------------
enum Modo { IDLE, ABERTA, FECHADA, VARREDURA, NEUTRO, E1, E2, E3 };
static Modo modo = IDLE;

Encoder enc(2, 3);
Servo servo;

static unsigned long t_prox, t_ant, t_ini;
static long ticks_ant;
static float setpoint = 0, pulso_cmd = 1500;
static float integ = 0;
static float v_ant_filt = 0;
static bool log_idle = false;

// filtros
static const uint8_t JMAX = 10;
static float jan[JMAX];
static uint8_t jn = 5, jidx = 0;
static float alfa = 0.3f;
static float v_lp = 0, v_mm = 0;
static uint8_t filtro_sel = 1;          // 0 = media movel, 1 = passa-baixa

// parametros dos ensaios/varredura
static float sw_passo = 50, sw_seg = 3;
static float e3_voltas = 3, e3_alvo_mm = 0;
static bool e3_parou = false;
static float e3_t_parada = 0;

// busca do neutro
static int neu_ini = 1400, neu_fim = 1900;
static int neu_i = -1;
static bool neu_marcado = false;
static long neu_tk0, neu_tk1;
static float neu_t0, neu_t1;
static int neu_par_min, neu_par_max;    // menor e maior pulso com roda parada

static float limita(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

static void escreve_pulso(float us) {
  pulso_cmd = limita(us, PULSO_MIN, PULSO_MAX);
  servo.writeMicroseconds((int)(pulso_cmd + 0.5f));
}

static void zera_filtros() {
  for (uint8_t i = 0; i < JMAX; i++) jan[i] = 0;
  jidx = 0; v_lp = 0; v_mm = 0; v_ant_filt = 0;
}

static void muda_setpoint(float sp) {
  if (sp != setpoint) integ = 0;          // zera a integral quando o setpoint muda
  setpoint = sp;
}

static void cabecalho() {
  Serial.println(F("t_s,ticks,s_mm,v_bruta,v_filtrada,setpoint,pulso_us,dt_us,v_mm,v_lp"));
}

static void inicia(Modo m) {
  enc.write(0);
  zera_filtros();
  integ = 0; setpoint = 0;
  modo = m;
  t_ant = micros();
  t_prox = t_ant + TS_US;
  t_ini = t_ant;
  ticks_ant = 0;
  e3_parou = false;
  if (m != NEUTRO) cabecalho();
}

static void para(const __FlashStringHelper *msg) {
  escreve_pulso(neutro_us);
  setpoint = 0; integ = 0;
  modo = IDLE;
  Serial.println(msg);
}

// fecha o passo da busca do neutro e imprime uma linha da tabela
static void fecha_passo_neutro() {
  int pulso = neu_ini + 10 * neu_i;
  long dtk = neu_tk1 - neu_tk0;
  float dts = neu_t1 - neu_t0;
  float v = dts > 0 ? K_MM_TICK * dtk / dts : 0;
  bool parada = labs(dtk) <= NEU_TICKS_PARADA;
  if (parada) {
    if (neu_par_min == 0 || pulso < neu_par_min) neu_par_min = pulso;
    if (pulso > neu_par_max) neu_par_max = pulso;
  }
  Serial.print(F("# TAB,")); Serial.print(pulso);
  Serial.print(','); Serial.print(dtk);
  Serial.print(','); Serial.print(v, 2);
  Serial.print(','); Serial.println(parada ? F("parada") : (dtk > 0 ? F("gira+") : F("gira-")));
}

static void amostra() {
  unsigned long agora = micros();
  long ticks = enc.read();
  float dt = (agora - t_ant) * 1e-6f;          // dt REAL medido
  unsigned long dt_us = agora - t_ant;
  t_ant = agora;
  float t = (agora - t_ini) * 1e-6f;

  float s = K_MM_TICK * ticks;
  float v = K_MM_TICK * (ticks - ticks_ant) / dt;     // bruta, mm/s
  ticks_ant = ticks;

  // dois filtros sempre calculados
  jan[jidx] = v; jidx = (jidx + 1) % jn;
  float soma = 0;
  for (uint8_t i = 0; i < jn; i++) soma += jan[i];
  v_mm = soma / jn;
  v_lp += alfa * (v - v_lp);
  float vf = filtro_sel == 0 ? v_mm : v_lp;

  // agenda dos ensaios
  switch (modo) {
    case NEUTRO: {
      int i = (int)(t / NEU_S);
      if (i != neu_i) {
        if (neu_i >= 0) fecha_passo_neutro();
        neu_i = i; neu_marcado = false;
        if (neu_ini + 10 * i > neu_fim) {
          if (neu_par_max > 0) {
            Serial.print(F("# parada de ")); Serial.print(neu_par_min);
            Serial.print(F(" a ")); Serial.print(neu_par_max);
            Serial.print(F(" us; banda morta = ")); Serial.print(neu_par_max - neu_par_min + 10);
            Serial.print(F(" us; NEUTRO = ")); Serial.print((neu_par_min + neu_par_max) / 2.0f, 1);
            Serial.println(F(" us"));
          } else Serial.println(F("# nenhuma faixa parada encontrada; amplie ini/fim"));
          para(F("# fim busca do neutro")); return;
        }
        escreve_pulso(neu_ini + 10 * i);
      }
      float tin = t - i * NEU_S;
      if (!neu_marcado && tin >= NEU_DESC_S) { neu_tk0 = ticks; neu_t0 = t; neu_marcado = true; }
      neu_tk1 = ticks; neu_t1 = t;
      return;                                    // sem linha de CSV: so a tabela
    }
    case E1: {
      int i = (int)(t / PAT_S);
      if (i >= 5) { para(F("# fim E1")); return; }
      muda_setpoint(pat[i]);
    } break;
    case E2:
      if (t >= 2 * INV_S) { para(F("# fim E2")); return; }
      muda_setpoint(t < INV_S ? v_ens : -v_ens);
      break;
    case E3:
      if (!e3_parou && fabs(s) >= e3_alvo_mm) {
        e3_parou = true; e3_t_parada = t; muda_setpoint(0);
        integ = 0;
      }
      if (e3_parou && t - e3_t_parada >= PARADA_S) {
        Serial.print(F("# s_final_mm=")); Serial.println(s, 2);
        Serial.print(F("# alvo_mm=")); Serial.println(e3_alvo_mm, 2);
        para(F("# fim E3")); return;
      }
      if (t > TMAX_E3) { para(F("# E3 estourou o tempo")); return; }
      if (!e3_parou) muda_setpoint(v_ens);
      break;
    default: break;
  }

  float pulso = pulso_cmd;
  if (modo == ABERTA) {
    // pulso fixo, ja escrito
  } else if (modo == VARREDURA) {
    int i = (int)(t / sw_seg);
    float p = PULSO_MIN + sw_passo * i;
    if (p > PULSO_MAX) { para(F("# fim varredura")); return; }
    escreve_pulso(p);
    pulso = pulso_cmd;
  } else if (modo == FECHADA || modo == E1 || modo == E2 || modo == E3) {
    float e = setpoint - vf;
    // integral limitada: a contribuicao ki*integ nao passa de +-250 us
    integ += e * dt;
    if (ki > 0) integ = limita(integ, -LIM_CORR_US / ki, LIM_CORR_US / ki);
    float dv = (vf - v_ant_filt) / dt;               // derivada na medida filtrada
    float u = kp * e + ki * integ - kd * dv;
    u = limita(u, -LIM_CORR_US, LIM_CORR_US);
    escreve_pulso(neutro_us + u);
    pulso = pulso_cmd;
  }
  v_ant_filt = vf;

  if (modo != IDLE || log_idle) {
    Serial.print(t, 3);       Serial.print(',');
    Serial.print(ticks);      Serial.print(',');
    Serial.print(s, 2);       Serial.print(',');
    Serial.print(v, 2);       Serial.print(',');
    Serial.print(vf, 2);      Serial.print(',');
    Serial.print(setpoint, 2);Serial.print(',');
    Serial.print(pulso, 0);   Serial.print(',');
    Serial.print(dt_us);      Serial.print(',');
    Serial.print(v_mm, 2);    Serial.print(',');
    Serial.println(v_lp, 2);
  }
}

static void params() {
  Serial.print(F("# D_mm=")); Serial.print(D_MM, 2);
  Serial.print(F(" N=")); Serial.print(N_TICKS, 1);
  Serial.print(F(" k_mm_tick=")); Serial.print(K_MM_TICK, 5);
  Serial.print(F(" neutro_us=")); Serial.print(neutro_us, 0);
  Serial.print(F(" pino_servo=")); Serial.print(pin_servo);
  Serial.print(F(" Ts_ms=")); Serial.println(TS_US / 1000.0f, 1);
  Serial.print(F("# kp=")); Serial.print(kp, 4);
  Serial.print(F(" ki=")); Serial.print(ki, 4);
  Serial.print(F(" kd=")); Serial.print(kd, 4);
  Serial.print(F(" filtro=")); Serial.print(filtro_sel == 0 ? F("media_movel") : F("passa_baixa"));
  Serial.print(F(" janela=")); Serial.print(jn);
  Serial.print(F(" alfa=")); Serial.println(alfa, 3);
  Serial.print(F("# patamares="));
  for (uint8_t i = 0; i < 5; i++) { Serial.print(pat[i], 1); Serial.print(' '); }
  Serial.print(F(" v_ens=")); Serial.println(v_ens, 1);
}

static void ajuda() {
  Serial.println(F("# h ? | n us | o us | s mm/s | x | z | g kp ki kd | f 0/1 | w n | a alfa"));
  Serial.println(F("# q v1..v5 | r mm/s | u pino | l 0/1 | N [ini] [fim] | A [passo] [s] | 1 | 2 | 3 [voltas]"));
}

static void comando(char *linha) {
  char c = linha[0];
  char *p = linha + 1;
  float a[5]; uint8_t n = 0;
  while (n < 5) {
    char *fim;
    float x = strtod(p, &fim);
    if (fim == p) break;
    a[n++] = x; p = fim;
  }
  switch (c) {
    case 'h': ajuda(); break;
    case '?': params(); break;
    case 'n': if (n) { neutro_us = limita(a[0], 1300, 1700); escreve_pulso(neutro_us); params(); } break;
    case 'o': if (n) { inicia(ABERTA); escreve_pulso(a[0]); } break;
    case 's':
      if (n) {
        if (modo != FECHADA) { inicia(FECHADA); }
        muda_setpoint(a[0]);
      }
      break;
    case 'x': para(F("# parado")); break;
    case 'z': enc.write(0); ticks_ant = 0; integ = 0; Serial.println(F("# zerado")); break;
    case 'g': if (n >= 3) { kp = a[0]; ki = a[1]; kd = a[2]; integ = 0; params(); } break;
    case 'f': if (n) { filtro_sel = a[0] != 0; params(); } break;
    case 'w': if (n) { jn = limita(a[0], 1, JMAX); params(); } break;
    case 'a': if (n) { alfa = limita(a[0], 0.01f, 1.0f); params(); } break;
    case 'q': if (n == 5) { for (uint8_t i = 0; i < 5; i++) pat[i] = a[i]; params(); } break;
    case 'r': if (n) { v_ens = fabs(a[0]); params(); } break;
    case 'u':
      if (n) {
        servo.detach(); pin_servo = (uint8_t)a[0];
        servo.attach(pin_servo, 1000, 2000); escreve_pulso(neutro_us);
        Serial.print(F("# servo no pino ")); Serial.println(pin_servo);
      }
      break;
    case 'l': if (n) log_idle = a[0] != 0; break;
    case 'N':
      neu_ini = n >= 1 ? (int)a[0] : 1400;
      neu_fim = n >= 2 ? (int)a[1] : 1900;
      neu_i = -1; neu_par_min = 0; neu_par_max = 0;
      inicia(NEUTRO);
      Serial.print(F("# busca do neutro de ")); Serial.print(neu_ini);
      Serial.print(F(" a ")); Serial.print(neu_fim);
      Serial.println(F(" us, 10 us por passo, 5 s cada"));
      Serial.println(F("# TAB,pulso_us,ticks_em_4s,v_mm_s,estado"));
      break;
    case 'A':
      sw_passo = n >= 1 ? a[0] : 50; sw_seg = n >= 2 ? a[1] : 3;
      inicia(VARREDURA);
      break;
    case '1': inicia(E1); break;
    case '2': inicia(E2); break;
    case '3':
      e3_voltas = n ? a[0] : 3;
      e3_alvo_mm = e3_voltas * PI * D_MM;
      inicia(E3);
      break;
    default: ajuda();
  }
}

void setup() {
  Serial.begin(115200);
  servo.attach(pin_servo, 1000, 2000);
  escreve_pulso(neutro_us);
  t_ant = micros(); t_prox = t_ant + TS_US;
  Serial.println(F("# TP4 controle de velocidade"));
  if (neutro_us == 1500.0f) Serial.println(F("# AVISO: neutro ainda e o 1500 padrao; faca a busca (comando N)"));
  params();
  ajuda();
}

void loop() {
  static char buf[48]; static uint8_t len = 0;
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (len) { buf[len] = 0; comando(buf); len = 0; }
    } else if (len < sizeof(buf) - 1) buf[len++] = ch;
  }
  if ((long)(micros() - t_prox) >= 0) {
    t_prox += TS_US;
    amostra();
  }
}
