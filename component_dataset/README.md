# A–F merged-camera detector dataset

This is a six-class A–F workspace seeded with the user's verified A–E dataset: 294 labeled training images (including its supplied augmentations) and 11 held-out validation images. Its original class IDs 0–4 are preserved. The six supplied close-up pictures remain in `references/` only. `inbox/Merged_photo.png` is the supplied merged top view; its visible A/E/F instances have been labeled and a set of synthetic lighting/paint variants can be generated from it.

**Important limitation:** the 240 synthetic variations and 120 zoom crops all come from one merged frame and belong in training only. They are not independent views or validation data. The held-out set contains no F examples and does not represent the merged-camera view. The current checkpoint was visually checked on the source frame only; it has not established accuracy across other positions, lighting, or paint conditions. Collect real merged frames in separate sessions and label them for a held-out validation set before production use.

Current experimental weights are at `runs/af_detector/weights/best.pt`; the camera app copy is `../best.pt` (the previous app model is backed up as `../best_before_corrected_AEF.pt`). The corrected annotation is at `annotation_preview_corrected.png`; the filtered source-frame check is at `runs/detect/final_candidate_check/Merged_photo.jpg`. Using the camera app's duplicate-suppression rule, the candidate returns A=5, E=2, F=10 on the supplied merged frame. This is a source-frame check, not proof of performance on other scenes; the available validation set still has no F examples and no merged-view examples.

Class IDs are fixed: `0=A`, `1=B`, `2=C`, `3=D`, `4=E`, `5=F`. Keep that ordering when labeling or training.

## Capture and label

1. Run the updated `combined_Cameras_.py` with both cameras connected.
2. Put the product in different positions and orientations, including small and large product sizes and the range of normal lighting. Capture frames with **SAVE TRAINING FRAME**. They are saved under `component_dataset/inbox/` beside the camera script.
3. Install the labeling tool requirements with `py -m pip install pillow` if Pillow is missing.
4. Run `py label_frames.py` from this folder. Select a class, draw a tight rectangle around every visible instance of that component, and choose **Save to TRAIN** or **Save to VALIDATION**. Keys 1–6 select A–F; Backspace removes the last box. Never put frames from the same short video burst in both splits.
5. Label every visible A–F object in a frame; do not leave visible components unlabelled. Use an EMPTY button for a true negative frame with no visible target.

For paint robustness, deliberately collect balanced real merged frames of A and E in each condition: no paint, light paint, partial paint, heavy paint, and fully painted while the component's outline/shape is still visible. Include different paint colors/thicknesses, shadows, glare, dirt, and product positions/sizes. Make sure every paint condition appears in both training and held-out validation frames; frames from the same short recording belong in only one split. Label A/E whenever their identity is still visually discernible, even if their surface is fully painted. If a painted mark/region has no component, or the part is fully covered and has no visible shape/edge evidence, use **Save EMPTY TRAIN** or **Save EMPTY VAL**. A normal RGB camera cannot identify a part from pixels where no identifying visual evidence remains.

The current product may contain only A, E, and F. Keep A–F as the model classes as requested; use product frames with B/C/D absent as hard negatives, and include labeled B/C/D examples from other products or camera views if those classes must also be detected. A reasonable first training attempt needs at least 30 distinct training frames and 8 separate validation frames, with at least 20 labeled instances of each class in training and 3 in validation. For dependable production inspection, gather substantially more (ideally hundreds of varied instances per class) and validate on different products/runs. These minimums only let the training script start; they do not guarantee accuracy.

## Train

The supplied merged image is annotated at `inbox/Merged_photo.txt`: A is the four horizontal perforated bars plus the narrow lower-center vertical strip, E is the two small keyhole-shaped parts between the bars, and F is the ten round silver studs. The corrected annotation is also reflected in the synthetic training labels. The current live-app BOM is A=5, B=0, C=0, D=0, E=2, F=10. To create lighting-aware training examples, run `py augment_merged.py inbox/Merged_photo.png --count 240 --seed 610171`; this set has already been generated in `images/train` and logged in `synthetic_generation_610171.txt`. It simulates dim/bright exposure, contrast and color shifts, shadows, blur, soft local glare, and paint-like overlays inside A/E boxes. These are correlated synthetic training examples only; never move them into validation. Synthetic glare does not replace real bright-camera captures, and it cannot restore a component whose shape has been erased by fully clipped white pixels.

From this folder, run `py -m pip install ultralytics pillow pyyaml` and then `py train_model.py`. Training fine-tunes the small pretrained YOLO11 model for the A–F classes. The first run may need an internet connection to download its starting weights. CPU training is supported but can take a long time; a supported NVIDIA GPU will be faster. The script warns when a class lacks validation examples; this workspace currently lacks F validation and independent merged-view validation.

The trainer uses broader HSV value augmentation (`hsv_v=0.70`) alongside the generated lighting variants. On 2026-10-17, the project-root `best.pt` was fine-tuned for 5 epochs at 640px from the previous `best.pt`; the run is saved at `runs/af_detector_lighting_retrain_20261017/`, and the previous app weights are backed up as `../best_before_lighting_retrain_20261017.pt`. The overall validation mAP50 was 0.909, but validation has no F examples, so that score measures A–E only and does not confirm F accuracy. Capture real F instances under bright/glare conditions in separate recording sessions and label some for validation before treating F performance as established. Full clipping to featureless white cannot be recovered by augmentation or model training.

## YOLO labels

Each image has a same-name text file in the matching `labels/train` or `labels/val` directory. One line per object: `class_id x_center y_center width height`; coordinates are normalized from 0 to 1. The label tool writes this format.

