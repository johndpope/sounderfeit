
#include <Stk.h>
#include <Bowed.h>

#include <cstdio>
#include <cmath>
#include <sys/time.h>

using namespace stk;

const int PRESSURE = 2;
const int POSITION = 4;
const int VELOCITY = 100;
const int FREQUENCY = 101;
const int VOLUME = 128;

// This frequency and cycle length selected after visual inspection
// based on findGoodPitch().
const StkFloat freq = 476.5;
const int L=201;

void findGoodPitch()
{
  // Create a Bowed instrument
  Bowed instr;
  instr.controlChange(PRESSURE, 100);
  instr.controlChange(POSITION, 50);
  instr.controlChange(VELOCITY, 150);
  instr.controlChange(VOLUME, 110);

  // Start a note
  instr.noteOn(freq, 1.0);

  // Capture a buffer of a given size after one second of playing, to
  // ensure it is steady-state.
  StkFloat buffer[L];
  for (int i=0; i<48000; i++) {
    StkFloat sample = instr.tick();
    if (i >= 48000-L)
      buffer[i-48000+L] = sample;
  }

  // Compare the continued steady-state with the repeated buffer for a
  // few periods to visual inspect that it doesn't diverge.
  for (int i=0; i<5000; i++) {
    StkFloat sample = instr.tick();
    printf("%g, %g\n", sample, buffer[i%(L)]);
  }
}

// Return the RMS of the cycle
StkFloat getCycle(StkFloat *buffer, int size,
                  StkFloat pressure, StkFloat position,
                  StkFloat velocity, StkFloat volume)
{
  // Create a Bowed instrument
  Bowed instr;
  instr.controlChange(PRESSURE, pressure);
  instr.controlChange(POSITION, position);
  instr.controlChange(VELOCITY, velocity);
  instr.controlChange(VOLUME, volume);

  // Start a note
  instr.noteOn(freq, volume / 128.0);

  // Capture a buffer of a given size after one second of playing, to
  // ensure it is steady-state.
  float meansq = 0.0;
  for (int i=0; i<48000; i++) {
    StkFloat sample = instr.tick();
    if (i >= 48000-size) {
      buffer[i-48000+size] = sample;
      meansq += sample*sample;
    }
  }
  meansq /= size;
  //fprintf(stderr, "meansq: %g\n", meansq);
  return sqrt(meansq);
}

void outputSingleCycle(StkFloat *buffer, int size, StkFloat scale)
{
  printf("%g", buffer[0] * scale);
  for (int i=1; i<size; i++)
    printf(",%g", buffer[i] * scale);
  printf("\n");
}

void iterateParameters()
{
  StkFloat minPressure = 0.0;
  StkFloat maxPressure = 128.0;
  StkFloat stpPressure = 1.0;
  StkFloat minPosition = 0.0;
  StkFloat maxPosition = 128.0;
  StkFloat stpPosition = 1.0;
  StkFloat minVelocity = 100.0;
  StkFloat maxVelocity = 100.0;
  StkFloat stpVelocity = 1.0;
  StkFloat minVolume = 100.0;
  StkFloat maxVolume = 100.0;
  StkFloat stpVolume = 1.0;
  StkFloat buffer[L];

  StkFloat pressure = minPressure;
  while (pressure <= maxPressure)
  {
    fprintf(stderr, "pressure: %g\n", pressure);
    StkFloat position = minPosition;
    while (position <= maxPosition)
    {
      StkFloat velocity = minVelocity;
      while (velocity <= maxVelocity)
      {
        StkFloat volume = minVolume;
        while (volume <= maxVolume)
        {
          StkFloat rms =
            getCycle(buffer, L, pressure, position, velocity, volume);
          if (rms > 1e-5) {
            printf("%g,%g,%g,%g,", pressure, position, velocity, volume);
            outputSingleCycle(buffer, L, 1.0);
          }
          volume += stpVolume;
        }
        velocity += stpVelocity;
      }
      position += stpPosition;
    }
    pressure += stpPressure;
  }
}

void sweepPressureAndPosition()
{
  StkFloat pressure=0, position=0.5, sample=0;
  double start=0, time=0;
  double lfo = 2;
  int sec=238;
  int period=201;
  int k=0;

  // Create a Bowed instrument
  Bowed instr;
  instr.controlChange(PRESSURE, pressure*64+64);
  instr.controlChange(POSITION, position*32+32);
  instr.controlChange(VELOCITY, 100.0);
  instr.controlChange(VOLUME, 100.0);

  // Start a note
  instr.noteOn(freq, 100.0 / 128.0);

  // Warm-up period 1 second
  for (int i=0; i<sec; i++) {
    for (int j=0; j<period; j++) {
      time = k / 48000.0;
      sample = instr.tick();
      k++;
    }
  }

  // Vary parameters while outputing them with a sample of the synth
  start=time;
  for (int i=0; i<sec; i++) {
    for (int j=0; j<period; j++) {
      time = k / 48000.0;
      pressure = sin((time - start) * 2 * M_PI * lfo)
        * exp(-(time-start-0.5)*(time-start-0.5) / 0.04);
      instr.controlChange(PRESSURE, pressure*64+64);
      instr.controlChange(POSITION, position*32+32);
      sample = instr.tick();
      printf("%g, %g, %g\n", pressure*64+64, position*32+32, sample);
      k++;
    }
  }
  start=time;
  for (int i=0; i<sec; i++) {
    for (int j=0; j<period; j++) {
      time = k / 48000.0;
      position = cos((time - start) * 2 * M_PI * lfo/4)/2;
      instr.controlChange(PRESSURE, pressure*64+64);
      instr.controlChange(POSITION, position*32+32);
      sample = instr.tick();
      printf("%g, %g, %g\n", pressure*64+64, position*32+32, sample);
      k++;
    }
  }
  start=time;
  for (int i=0; i<sec; i++) {
    for (int j=0; j<period; j++) {
      time = k / 48000.0;
      position = -cos((time - start) * 2 * M_PI * lfo/4)/2;
      pressure = sin((time - start) * 2 * M_PI * lfo/2)
        * exp(-(time-start-0.5)*(time-start-0.5) / 0.04);
      instr.controlChange(PRESSURE, pressure*64+64);
      instr.controlChange(POSITION, position*32+32);
      sample = instr.tick();
      printf("%g, %g, %g\n", pressure*64+64, position*32+32, sample);
      k++;
    }
  }
}


// Randomly slide parameters, sample a single buffer every now and then
StkFloat randomWalkParameters(int how_many)
{
  StkFloat volume = 100.0;
  StkFloat velocity = 100.0;
  StkFloat pressure = 64.0;
  StkFloat position = 64.0;

  const int size = L;
  StkFloat buffer[L];

  struct timeval t;
  gettimeofday(&t, NULL);
  srand(t.tv_usec);

  // Create a Bowed instrument
  Bowed instr;
  instr.controlChange(PRESSURE, pressure);
  instr.controlChange(POSITION, position);
  instr.controlChange(VELOCITY, velocity);
  instr.controlChange(VOLUME, volume);

  // Start a note
  instr.noteOn(freq, volume / 128.0);

  for (int n = 0; n < how_many; n++)
  {
    // Set pressure and position randomly
    pressure = ((double)rand()) / RAND_MAX * 128.0;
    position = ((double)rand()) / RAND_MAX * 128.0;
    instr.controlChange(PRESSURE, pressure);
    instr.controlChange(POSITION, position);

    // Wait random amount of time
    int r = ((double)rand()) / RAND_MAX * 48000.0/2;
    for (int k=0; k<r; k++)
      instr.tick();

    // Capture a buffer of a given size after one second of playing, to
    // ensure it is steady-state.
    float meansq = 0.0;
    for (int i=0; i<size; i++) {
      StkFloat sample = instr.tick();
      buffer[i] = sample;
      meansq += sample*sample;
    }
    double rms = sqrt(meansq  / size);

    if (rms > 1e-3) {
      // Display the parameters and buffer on a single line
      printf("%g,%g,%g,%g", pressure, position, velocity, volume);
      for (int i = 0; i < size; i++)
        printf(",%g", buffer[i]);
      printf("\n");
    };
  }
}

int main()
{
  Stk::setSampleRate(48000);

  //findGoodPitch();
  //iterateParameters();
  //sweepPressureAndPosition();
  randomWalkParameters(1000000);
}
