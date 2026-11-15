# The Fly Band, by Ear: what is happening and what is modeled

14 November 2026. This is a copy of the writeup doc; the doc also has the pipeline as a drawing.

## In short

The page splits an uploaded song into drums, bass, vocals and keys, and plays each part to its own simulated fruit fly. Each fly has a modeled pair of antennae, a copy of one real fly brain's wiring, and a stand-in wing. You can listen to four things per fly: the part as played, the movement of its antennae, the song its output neurons would sing, and a recreation of the part rebuilt from the moments its brain could not tell apart.

The ear and the brain are modeled on published measurements and data. The wing and the recreation rule are inventions, and so is the premise: real female flies listen to song, they do not sing it back.

## What happens when you press start

The song is cut down, split in four, and each part goes through one fly.

```
Your song -> Excerpt (liveliest 30 s, on the beat) -> Split (ordinary signal processing)
                                                          |
          Four parts: drums, bass, vocals, keys  ---------+----> hear: "Your song"
                       | one part to each fly, loud
   inside each fly:  Antennae -> Brain (8,161 neurons) -> Output neurons -> Stand-in wing
                        |            |                                          |
            hear: "Their antennae"   | chooses which moments to play   hear: "Their songs"
                             hear: "Their recreation"
```

1. The file is decoded and its liveliest 30 seconds are chosen, cut on the beat.
2. That excerpt is split into drums, bass, vocals and keys. This step is not the fly.
3. Each part is played to one fly. Its antennae swing, its antennal neurons fire, activity spreads through the brain to the output neurons, and the stand-in wing sings.
4. Each fly's brain activity is used to rebuild its part from other moments of itself.
5. The four flies run at once in background threads. You hear your song while they listen, and the page slides into the recreation when all four are ready.

## What is modeled, and how faithfully

The ear, the wiring and the neurons come from published work; the wing, the split and the recreation rule do not.

| Piece | Based on | Simplified or invented |
| --- | --- | --- |
| Antenna | Measured mechanics: one moderately damped resonator (Q 1.24) whose best frequency is about 800 Hz for loud sound and falls about 11 Hz per dB toward 200 Hz for faint sound, which it also amplifies | Playback loudness is assumed. Retuning and boost follow a smoothed loudness, not the real active process |
| Antennal neurons | The five real classes on each side, taken from the FlyWire annotations. A and B sense vibration (B below about 100 Hz, A above). C, D and E sense being pushed, and join in once the swing passes about 200 nm | The curves from movement to firing rate are mine. Which half of A and B fires on which direction of swing is random |
| Brain wiring | One female fly's connectome (FlyWire v783), as packaged with the Shiu et al. model: 138,639 neurons, 15.1 million connections | Only the 8,161 neurons that sound can affect are simulated. In tests this gave spike-for-spike the same result as the full brain |
| Neurons | The Shiu et al. leaky integrate-and-fire model with its published constants | Every neuron is identical apart from its wiring. No learning, no background activity, no chemistry beyond excite or inhibit |
| Output | The brain's descending neurons, its only wires to the body. Sound reaches 255 of the 1,299 | None |
| Wing and song | The elements of real fly song: clicks about 35 ms apart, a hum near 150 Hz, clicks as the louder mode, a short hum after a click train | The circuit is invented. The real one is in the body, outside this data set. Four settings were tuned on a test song |
| Split into four parts | Not the fly | Ordinary signal processing |
| Recreation | Not biology | An invented rule that uses the brain's activity and nothing else |

## The ear

A fly hears with its antennae: sound moves a feathered arm, and that movement stretches neurons at its base. The model follows that chain for each antenna, with the left audio channel moving the left antenna.

1. **Loudness.** Each part is played so its loud moments move the air at 1.5 mm/s, roughly 90 dB. A quiet part is boosted by at most four times its level in the mix.
2. **The antenna.** One resonator per antenna. Its best frequency is 800 Hz when the air moves at 1 mm/s or more and drops 11 Hz per dB below that, down to 200 Hz. Faint sound is boosted by up to ten times. The result is the antenna's swing in nanometres. Almost nothing above 800 Hz survives.
3. **Vibration neurons (classes A and B).** Class B follows the swing below about 100 Hz, class A the swing above. They start firing at 5 nm, reach half rate at 80 nm more, and top out at 260 spikes a second. Half of each class fires on each direction of swing.
4. **Deflection neurons (classes C, D and E).** These normally sense wind and gravity. C and D fire when the antenna is pushed forward, E when it is pushed back, and only once the swing passes 200 nm. They are silent for quiet sound and join in on loud passages.

| Class | Left | Right |
| --- | --- | --- |
| A (fast vibration) | 48 | 51 |
| B (slow vibration) | 156 | 103 |
| C (pushed forward) | 56 | 40 |
| D (pushed forward) | 28 | 25 |
| E (pushed back) | 173 | 164 |

## The brain

Each fly's brain is the same wiring diagram, run with a different random seed. The antennal neurons fire at the rates the ear gives them, and every other neuron fires only when its inputs push it over threshold.

- **Neuron model.** Rest at -52 mV, fire at -45 mV, reset to -52 mV. Membrane time constant 20 ms, synapse 5 ms, 2.2 ms refractory, 1.8 ms delay, 0.275 mV per synapse, time step 0.1 ms. These are the published constants.
- **How much of the brain sound reaches.** Driving all five antennal classes hard and in many patterns made 2,496 neurons fire at least once, under 2% of the brain. In a normal run 500 to 1,200 fire for one part.
- **The network that ships.** Those 2,496 plus every neuron receiving 30 or more excitatory synapses from them: 8,161 neurons and 857,264 connections. On three held-out random stimuli it produced exactly the same spikes as the full brain, neuron by neuron.
- **Output.** 255 of the 1,299 descending neurons can be reached. In a run, 20 to 55 are active. Left antenna drives mostly left-side outputs and right drives right, by three to one or more.
- **Named neurons.** The giant fibre, the escape neuron, fires readily. The neurons known to carry a female's response to courtship song (vpoEN, vpoDN, pC1) are never reached by sound in this model.

## The song

"Their songs" is what each fly would sing if its output neurons drove a wing. It contains no sound from the record.

- **Drive.** Spikes from the left-side and right-side output neurons are counted each millisecond and smoothed over 20 ms.
- **Which wing.** One wing sings at a time: the side whose outputs are busier. It switches when the other side has been 1.3 times busier for 40 ms.
- **Clicks.** When the drive is 2.5 times its own average over the last 1.5 seconds, the wing clicks every 35 ms. Each click is a 280 Hz pulse about 5 ms long, as loud as the drive.
- **Hum.** Otherwise the wing hums at 150 Hz, at about a third of the loudness a click would have. A short hum also follows each click train.
- **Loudness.** A drive of 0.6 spikes per millisecond counts as a full-strength wing beat.

The result follows the rhythm and loudness of the part. It cannot carry melody: fly song has two sounds.

## The recreation

"Their recreation" rebuilds each part from other moments of itself, chosen by the fly's brain. Nothing plays back in order and nothing is fitted to the music.

1. The part is cut into sixteenth-note slots on the detected beat (156 ms each at 96 bpm).
2. For each slot, the brain's state is recorded: how often each neuron fired during it, in three steps, counting only the brain's own neurons and not the antennal ones.
3. Each slot is replaced by a different slot at least a beat away: the one whose brain state is closest.
4. The chosen slots are laid end to end with 6 ms cross-fades.

Where the result sounds right, the fly's brain can tell those moments apart. Where it sounds wrong, the two moments looked alike to it. A fly has no sense of pitch above its antenna's range, so drums and bass come through better than vocals.

The percentage under each fly measures this. It compares how alike the swapped-in slots sound to the ones they replace (loudness in 16 pitch bands) against swapping at random: 0% is chance, 100% is identical.

## The split into four parts

The split is ordinary signal processing, done before any fly is involved. It looks at the song's spectrogram three times, each with a window length suited to one question, then shares the result out by pitch and stereo position.

| Pass | Window | Question | Goes to |
| --- | --- | --- | --- |
| 1 | 186 ms | Above about 200 Hz, what holds its pitch for half a second? | Held sound |
| Low | 93 ms | Below about 300 Hz, does the sound keep its level for a few tenths of a second, or die away? | Bass if it holds, kick drum if it dies away |
| 2 | 23 ms | In what is left, what changes smoothly from instant to instant, and what is sudden? | Smooth sound, or drums |

Held and smooth sound is then shared out. Low and in the centre of the stereo picture is bass. Centred and between 150 Hz and 6 kHz is vocals. The rest is keys. Unheld sound above about 6 kHz is treated as cymbals. In a mono file, vocals are simply the smooth sound in the voice range and keys are nearly empty.

Measured on a synthetic test song with a voice-like lead (vibrato, glides, consonants) and a tight kick, where the true parts are known:

| Part | What it is made of | Earlier one-pass split |
| --- | --- | --- |
| Drums | 88% drums, 11% bass, 1% vocals | 79% drums, 10% bass, 6% vocals, 6% keys |
| Bass | 90% bass, 5% drums, 5% keys | 66% bass, 31% drums |
| Vocals | 84% vocals, 14% bass | 81% vocals, 17% bass |
| Keys | 67% keys, 19% bass, 12% drums | 38% keys, 33% drums, 27% bass |

With a boomy kick that rings for a few tenths of a second, the bass part falls to 60% bass and 37% kick.

## Checks that were run

All checks used synthetic test songs made for the purpose. No real track has been run and nobody has listened to the output.

| Check | Result |
| --- | --- |
| Does the wiring matter? Same neurons and connection strengths, every connection moved to a random neuron | "Tells moments apart" fell from 39 / 15 / 4 / 15% (drums / bass / vocals / keys) to 8 / 8 / 1 / 0%. Active output neurons fell from 27 to 49 per fly to 6 to 14 |
| Is the small network the same as the full brain? | Identical spike counts for every neuron on three held-out one-second stimuli |
| Is the recreation just playback? | Its waveform had no correlation with the song (-0.04) while its loudness followed the song at 0.84 |
| Does each listening mode play what it says? Recorded from the page's sound output | "Your song" matched the uploaded file. A replay was identical. A soloed fly plus the other three equalled the full mix |
| Does the split work? | See the table above |
| Does the folder run? | The separate-files version over a local server and the single-file build both ran start to finish in a headless browser with no errors |

The vocals fly scores close to chance in every run, which is the expected result for an ear with no sense of pitch.

## Limits

**The premise is invented.** This brain is female, and female flies do not sing back. Her real answer to a courting male's song is to accept or reject him, through a known pair of descending neurons that sound never reaches in this model.

**The model is thin.**

- Every neuron is the same simple unit. There is no learning, no background activity and no modulation by the fly's state.
- In runs on a full mix, the giant fibre fired 20 to 30 times a second. A real fly would have jumped away at the first spike.
- This brain's right-side outputs respond about four times more strongly than its left, so the songs sit mostly on the right wing. It is one individual's wiring, and the two sides were not mapped equally.
- In an earlier check, my port of the neuron model fired about 20% more than the authors' reference output. I did not resolve that.

**The ear rests on an assumed loudness.** How many neurons fire, and whether the deflection classes join in, depends on the 90 dB assumption.

**The split is rule-based.** Bass notes in the voice range land in the vocals, long kicks land partly in the bass, and keys are whatever is left. Reverb, backing vocals and guitars on real records will blur it further.

**The slots follow the detected beat.** If beat detection is off, cuts land off the beat and the recreation sounds choppier.

**Sound quality.** Everything is processed at 22 kHz, so the parts sound duller than the original file.

## Folder map

See [README.md](README.md).

## Sources

| Used for | Source |
| --- | --- |
| Neuron model and connectivity tables | [Shiu et al., Drosophila brain model (code and data)](https://github.com/philshiu/Drosophila_brain_model) and its [paper](https://www.biorxiv.org/content/10.1101/2023.05.02.539144v1) |
| Neuron classes, sides and positions | [FlyWire annotations](https://github.com/flyconnectome/flywire_annotations) |
| The five classes of antennal neurons and what each senses | [Ishikawa et al., Stereotyped Combination of Hearing and Wind/Gravity-Sensing Neurons in the Johnston's Organ](https://www.frontiersin.org/articles/10.3389/fphys.2019.01552/abstract) |
| Deflection classes joining in above 200 nm; hearing range | [Albert and Göpfert, Hearing in Drosophila](https://pmc.ncbi.nlm.nih.gov/articles/PMC4582067) |
| Antenna as a damped resonator, Q 1.24 | [Göpfert and Robert, The mechanical basis of Drosophila audition](https://cob.silverchair.com/jeb/article-pdf/205/9/1199/1243755/1199.pdf) |
| Best frequency 800 Hz falling 11 Hz per dB | [Nadrowski and Göpfert, Transducer-based active processes in hearing and best-frequency shifts](https://pmc.ncbi.nlm.nih.gov/articles/PMC2649288) |
| Clicks and hum in real song, and what drives them | [Roemschied et al., Flexible circuit mechanisms for context-dependent song sequencing](https://link.springer.com/10.1038/s41586-023-06632-1) |
| How a female fly actually responds to song | [Wang et al., Neural circuit mechanisms of sexual receptivity in Drosophila females](https://www.nature.com/articles/s41586-020-2972-7) |

Figures for the antenna and the antennal neurons were taken from these pages as returned by web search; I did not read the full papers.
