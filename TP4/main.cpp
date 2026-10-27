// Conta os ticks do encoder girando a roda a mão (Arduino Mega 2560).
//
// Usa a biblioteca Encoder (lib/Encoder, da verlab/hero_common): canal A
// no pino 2 (INT4) e canal B no pino 3 (INT5). Com os dois pinos de
// interrupção ela conta as quatro bordas de cada ciclo (X4) e já liga os
// pull-ups internos. Se o sinal sair invertido, troque A e B.
//
// O motor não é comandado: sem pulso no servo, a roda gira livre a mão.
// Para medir N, zere com 'z' no monitor serial, dê uma volta na roda e
// leia o valor de ticks.

#include <Arduino.h>
#include <Encoder.h>
#define INPUT_PULLUP
#define ENCODER_USE_INTERRUPTS

Encoder enc(2, 3);

void setup() {
  Serial.begin(115200);
  Serial.println("# girando a mao; envie 'z' para zerar");
}

void loop() {
  static long anterior = 0;

  if (Serial.available() && Serial.read() == 'z') {
    enc.write(0);
    anterior = 0;
    Serial.println("ticks: 0  (zerado)");
  }

  long atual = enc.read();
  if (atual != anterior) {
    Serial.print("ticks: ");
    Serial.println(atual);
    anterior = atual;
  }
}
