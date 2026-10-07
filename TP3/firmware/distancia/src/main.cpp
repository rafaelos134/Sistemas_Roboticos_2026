// Sketch final do TP3: o TCRT5000 devolvendo distância em milímetros.
//
// Mesmo ciclo do roteiro (acende, espera, lê; apaga, espera, lê) e o
// sinal S = max(0, adc_off - adc_on). A distância sai do modelo escolhido
// em calibracao.h. Cada linha da serial é
//
//   sinal,mm
//
// formato que o exemplos/plot_serial.py desenha em milímetros. Linhas que
// começam com '#' são comentários e o plot_serial as ignora.
//
// Fora da faixa de ajuste o modelo não vale, e o programa imprime:
//   - D_MIN quando o fototransistor satura (cartão perto demais);
//   - D_DET quando S < S_LIM: o sinal não se separa do fundo, e a leitura
//     deve ser entendida como "o cartão está a D_DET ou mais". Isso
//     também evita a divisão por zero quando S = 0.

#include <Arduino.h>
#include <math.h>
#include "calibracao.h"

const int PIN_LED = 8;
const int PIN_ADC = A0;

void espera_us(unsigned long us) {
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

float d_roteiro(float s) { return G0 + G1 / sqrt(s); }

float d_potencia(float s) { return pow(ALFA / s, 1.0 / N_EXP); }

float d_tabela(float s) {
  if (s >= TAB_S[0]) return TAB_D[0];
  for (int i = 1; i < TAB_N; i++) {
    if (s >= TAB_S[i]) {
      float f = (s - TAB_S[i - 1]) / (TAB_S[i] - TAB_S[i - 1]);
      return TAB_D[i - 1] + f * (TAB_D[i] - TAB_D[i - 1]);
    }
  }
  return TAB_D[TAB_N - 1];
}

float modelo(float s) {
#if MODELO == 1
  return d_roteiro(s);
#elif MODELO == 2
  return d_potencia(s);
#else
  return d_tabela(s);
#endif
}

float distancia_mm(int adc_on, int sinal) {
  if (adc_on <= ADC_SAT) return D_MIN;
  if (sinal < S_LIM) return D_DET;
  return constrain(modelo(sinal), D_MIN, D_DET);
}

// Tempo de cálculo de cada modelo, para a tabela de comparação.
void mede_custo() {
  volatile float acc = 0;
  const int n = 1000;
  float (*f[3])(float) = {d_roteiro, d_potencia, d_tabela};
  for (int k = 0; k < 3; k++) {
    unsigned long t0 = micros();
    for (int i = 0; i < n; i++) acc += f[k](10.0 + (i % 500));
    unsigned long dt = micros() - t0;
    Serial.print("# custo M");
    Serial.print(k + 1);
    Serial.print(" = ");
    Serial.print((float)dt / n, 1);
    Serial.println(" us por amostra");
  }
}

void setup() {
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  Serial.begin(115200);
  Serial.print("# tcrt5000 distancia, modelo M");
  Serial.println(MODELO);
  mede_custo();
}

void loop() {
  digitalWrite(PIN_LED, HIGH);
  espera_us(T_ON_US);
  int adc_on = ler_media(M_CONV);

  digitalWrite(PIN_LED, LOW);
  espera_us(T_OFF_US);
  int adc_off = ler_media(M_CONV);

  int sinal = adc_off - adc_on;
  if (sinal < 0) sinal = 0;

  Serial.print(sinal);
  Serial.print(',');
  Serial.println(distancia_mm(adc_on, sinal), 1);
}
