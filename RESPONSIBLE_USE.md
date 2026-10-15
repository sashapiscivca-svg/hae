# Responsible use

[Українська](RESPONSIBLE_USE.uk.md)

hae exists to **find people who need help**: people swept away by floods and tsunamis, lost in forests and mountains, or stranded after earthquakes. We built it to run on very cheap hardware so that volunteer and under-funded rescue teams can use it too. We measure success in lives saved.

## What the licence says, and what we ask

The code and weights are released under the **MIT License**. It places no legal restriction on fields of use, and we will not pretend that it does. What follows is our **public request and commitment**, not a licence condition.

### We ask you not to use hae for

1. **Military purposes**: reconnaissance, detecting, tracking or targeting people for combat, fire control, target designation, or any system meant to inflict harm. This includes "defensive" and ISR uses.
2. **Persecution**: finding, tracking or pressuring people for their beliefs, religion, origin, language, sex, orientation, political or civic activity, or for taking part in assemblies and protests.
3. **Mass or covert surveillance**: continuous monitoring of populations, or tracking individuals without their knowledge and a lawful basis.
4. **Repression**: helping any actor suppress rights and freedoms, detain arbitrarily, displace people by force, or violate international humanitarian or human-rights law.
5. **Acting against people who need protection**: locating refugees or migrants in order to push them back or detain them unlawfully, or hunting people in distress instead of helping them.
6. **Identifying individuals**: face recognition, biometrics, linking a detection to an identity, or inferring group membership from appearance.
7. **Any use intended or foreseeably likely to cause death, injury, coercion or intimidation.**

If you cannot tell whether a use helps or harms, please treat it as out of scope.

### Uses we built it for

- Search and rescue of people; searching for missing persons at the request of rescue services or families.
- Disaster response: floods, tsunamis, earthquakes, wildfires, avalanches.
- Research and teaching consistent with the list above.
- Humanitarian missions overseen by independent organisations.

## How the design supports this

- **Presence and location only.** The model outputs "a person is probably here" as a box and a score. It has no identity, face, age, sex or other attribute outputs, and we will not add them.
- **Runs locally.** Inference is on-device. Nothing is sent to us or to any cloud.
- **A human decides.** hae suggests; a rescuer decides. Do not wire detections to automated actions against people.
- **Honest limits.** We publish miss rates, false-alarm rates and the conditions where the models fail (see the [model card](MODEL_CARD.md)). A missed person can mean a death, so we do not hide weaknesses.
- **Clean provenance.** Only civilian, openly licensed data; no third-party pretrained weights; no datasets of military origin or funding.

## Operational safety for rescue teams

- **Never call off or narrow a search because the detector found nothing.** At the recommended thresholds, the models still miss a substantial share of people (see the recall figures in the model card).
- Validate the model on your own camera and terrain before an operation, and keep the human visual search running.
- Confirm detections over several frames or passes before dispatching a team. Expect false alarms from warm rocks, animals, vehicles and debris.
- Follow aviation rules, your organisation's SAR protocols, and local privacy law on recording and storing footage of people.
- Fit a heatsink to a Raspberry Pi Zero 2 W running hae in flight. It throttles above about 80 °C.

## What we do

- We provide no support, consulting, hosting or partnership to projects that conflict with this document, and we will publicly disclose credible reports of such use.
- Changes that strengthen this document may be made at any time. Any weakening will be justified publicly and will take effect no sooner than 30 days after it is announced.

## Reporting

To report misuse, or a safety problem in the models (for example a systematic failure mode that could cost lives), follow [SECURITY.md](SECURITY.md). We read every report.
