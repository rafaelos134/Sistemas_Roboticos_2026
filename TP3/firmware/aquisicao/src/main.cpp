// Firmware de aquisição do TCRT5000 (TP3).
//
// Mesmo circuito do roteiro: LED infravermelho no D8 (via 100 ohm) e
// coletor do fototransistor no A0 (pull-up de 4,7 k). O ciclo também é
// o do roteiro: acende, espera t_on, lê; apaga, espera t_off, lê; o
// sinal é max(0, adc_off - adc_on). O que muda:
//
//   - cada linha traz os tempos e as duas leituras, não só o sinal;
//   - cada leitura é a média de M conversões no fim da janela;
//   - os tempos ficam em microssegundos e mudam por comando na serial,
//     sem regravar;
//   - com ciclos curtos, as medidas vão para um buffer e só depois são
//     impressas, para que a serial não alongue o tempo desligado;
//   - um modo "degrau" mede a resposta do A0 logo após o LED ligar e
//     desligar.
//
// Saída (115200 baud):
//   A,t_on_us,t_off_us,adc_on,adc_off,sinal   uma linha por ciclo
//   G,borda,t_us,a0                           degrau (borda 1 = liga, 0 = apaga)
//   # ...                                     comentários e confirmações
//
// Comandos (uma linha cada, terminada em \n):
//   on <us>          tempo ligado em microssegundos
//   off <us>         tempo desligado em microssegundos
//   m <M>            conversões por leitura (1 a 64)
//   pausa | go       para / retoma os ciclos (pausado, o LED fica apagado)
//   adc rapido       divisor do ADC em 16 (~16 us por conversão)
//   adc normal       divisor do ADC em 128 (~112 us, padrão do Arduino)
//   degrau <K> <R>   K amostras após cada borda, média de R repetições
//   ?                mostra a configuração

#include <Arduino.h>

const int PIN_LED = 8;
const int PIN_ADC = A0;

// Com t_on ou t_off abaixo disso, usa M = 1: as conversões extras
// alongariam a janela que está sendo estudada.
const uint32_t T_MEDIA_MIN_US = 5000;
// Ciclos mais curtos que isso são medidos em blocos e impressos depois.
const uint32_t T_BLOCO_US = 20000;
const int BLOCO = 100;
// Limites do modo degrau.
const int K_MAX = 250;
const int R_MAX = 50;

uint32_t t_on_us = 1000000UL;
uint32_t t_off_us = 1000000UL;
int m_conv = 8;
bool rodando = true;
bool adc_rapido = false;

int buf_on[BLOCO];
int buf_off[BLOCO];
uint32_t soma_a[K_MAX];
uint32_t soma_t[K_MAX];

char linha[40];
int n_linha = 0;

void espera_us(uint32_t us) {
  // delayMicroseconds só é exato até 16383 us.
  if (us >= 16000UL) {
    delay(us / 1000UL);
    delayMicroseconds(us % 1000UL);
  } else {
    delayMicroseconds(us);
  }
}

int ler_media(int m) {
  long s = 0;
  for (int i = 0; i < m; i++) s += analogRead(PIN_ADC);
  return (int)((s + m / 2) / m);
}

void ciclo(int m, int &adc_on, int &adc_off) {
  digitalWrite(PIN_LED, HIGH);
  espera_us(t_on_us);
  adc_on = ler_media(m);
  digitalWrite(PIN_LED, LOW);
  espera_us(t_off_us);
  adc_off = ler_media(m);
}

void imprime(int adc_on, int adc_off) {
  int sinal = adc_off - adc_on;
  if (sinal < 0) sinal = 0;
  Serial.print("A,");
  Serial.print(t_on_us);
  Serial.print(',');
  Serial.print(t_off_us);
  Serial.print(',');
  Serial.print(adc_on);
  Serial.print(',');
  Serial.print(adc_off);
  Serial.print(',');
  Serial.println(sinal);
}

int m_efetivo() {
  if (t_on_us < T_MEDIA_MIN_US || t_off_us < T_MEDIA_MIN_US) return 1;
  return m_conv;
}

void define_adc(bool rapido) {
  adc_rapido = rapido;
  ADCSRA = (ADCSRA & ~0x07) | (rapido ? 0x04 : 0x07);
  analogRead(PIN_ADC);  // a primeira conversão após a troca é descartada
}

void mostra_config() {
  Serial.print("# t_on_us=");
  Serial.print(t_on_us);
  Serial.print(" t_off_us=");
  Serial.print(t_off_us);
  Serial.print(" m=");
  Serial.print(m_conv);
  Serial.print(" m_efetivo=");
  Serial.print(m_efetivo());
  Serial.print(" adc=");
  Serial.print(adc_rapido ? "rapido" : "normal");
  Serial.print(" estado=");
  Serial.println(rodando ? "rodando" : "pausa");
}

void degrau(int k, int r) {
  if (k < 1 || k > K_MAX || r < 1 || r > R_MAX) {
    Serial.println("# erro: degrau <K 1..250> <R 1..50>");
    return;
  }
  for (int borda = 1; borda >= 0; borda--) {
    for (int i = 0; i < k; i++) soma_a[i] = soma_t[i] = 0;
    for (int rep = 0; rep < r; rep++) {
      // Estado oposto por 100 ms, para partir do regime.
      digitalWrite(PIN_LED, borda ? LOW : HIGH);
      delay(100);
      uint32_t t0 = micros();
      digitalWrite(PIN_LED, borda ? HIGH : LOW);
      for (int i = 0; i < k; i++) {
        uint32_t t = micros() - t0;
        soma_a[i] += analogRead(PIN_ADC);
        soma_t[i] += t;
      }
    }
    digitalWrite(PIN_LED, LOW);
    for (int i = 0; i < k; i++) {
      Serial.print("G,");
      Serial.print(borda);
      Serial.print(',');
      Serial.print((soma_t[i] + r / 2) / r);
      Serial.print(',');
      Serial.println((float)soma_a[i] / r, 1);
    }
  }
  Serial.println("#fim degrau");
}

void executa(char *cmd) {
  char *nome = strtok(cmd, " \t\r");
  if (nome == NULL) return;
  char *a1 = strtok(NULL, " \t\r");
  char *a2 = strtok(NULL, " \t\r");

  if (!strcmp(nome, "on") && a1) {
    t_on_us = strtoul(a1, NULL, 10);
  } else if (!strcmp(nome, "off") && a1) {
    t_off_us = strtoul(a1, NULL, 10);
  } else if (!strcmp(nome, "m") && a1) {
    m_conv = constrain(atoi(a1), 1, 64);
  } else if (!strcmp(nome, "pausa")) {
    rodando = false;
    digitalWrite(PIN_LED, LOW);
  } else if (!strcmp(nome, "go")) {
    rodando = true;
  } else if (!strcmp(nome, "adc") && a1) {
    define_adc(!strcmp(a1, "rapido"));
  } else if (!strcmp(nome, "degrau") && a1 && a2) {
    Serial.println("#ok degrau");
    degrau(atoi(a1), atoi(a2));
    return;
  } else if (!strcmp(nome, "?")) {
    mostra_config();
    return;
  } else {
    Serial.print("# erro: comando desconhecido: ");
    Serial.println(nome);
    return;
  }
  Serial.print("#ok ");
  Serial.println(nome);
  mostra_config();
}

void le_comandos() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      linha[n_linha] = '\0';
      executa(linha);
      n_linha = 0;
    } else if (n_linha < (int)sizeof(linha) - 1) {
      linha[n_linha++] = c;
    }
  }
}

void setup() {
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  Serial.begin(115200);
  define_adc(false);
  Serial.println("# tcrt5000 aquisicao");
  mostra_config();
}

void loop() {
  le_comandos();
  if (!rodando) return;

  int m = m_efetivo();
  int adc_on, adc_off;

  if (t_on_us + t_off_us >= T_BLOCO_US) {
    ciclo(m, adc_on, adc_off);
    imprime(adc_on, adc_off);
    return;
  }

  // Ciclo curto: um ciclo de aquecimento (o LED vinha apagado havia mais
  // tempo que t_off) e depois BLOCO ciclos seguidos, sem serial no meio.
  ciclo(m, adc_on, adc_off);
  for (int i = 0; i < BLOCO; i++) ciclo(m, buf_on[i], buf_off[i]);
  digitalWrite(PIN_LED, LOW);
  for (int i = 0; i < BLOCO; i++) imprime(buf_on[i], buf_off[i]);
}
