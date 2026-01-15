BME280 (I2C)   : SDA -> GPIO21    (default SDA)
                 SCL -> GPIO22    (default SCL)

PMS7003 (UART) : RX -> TX ESP32 (GPIO17)  
  		           TX -> RX ESP32 (GPIO16)
                 Power 5V/GND

MAX485 (RS485) : RO  -> GPIO32    (Serial1 RX)
                 DI  -> GPIO33    (Serial1 TX)
                 DE/RE -> GPIO13  (kontrol arah, output)
                 VCC -> 3.3V
                 GND -> common GND

Multiplexer HW-178:
  SIG (analog out) -> GPIO35
  S0 -> GPIO25
  S1 -> GPIO26
  S2 -> GPIO27
  S3 -> GPIO4

Anemometer power : 12V (supply terpisah) dan GND common
PMS7003 power     : 5V (pakai stepdown), GND common
Semua GND harus sama (common ground)
