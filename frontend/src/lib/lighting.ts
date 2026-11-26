// Scene brightness check for the live camera.
//
// experiments/results/*-cv_landmarks: MediaPipe lite keeps working in dim light (PCK@0.1 vs the
// heavy model: 0.89 clean, 0.83 at mean luminance 0.13) and drops at about 0.09 (0.68). Brightening
// the frame before detection (canvas gain, gamma + CLAHE) made landmarks WORSE at every level,
// because it amplifies sensor noise as much as signal. So FormFit does not enhance frames; it tells
// the user to add light when the scene is darker than DARK_LUMA.

export const DARK_LUMA = 0.11;

let canvas: HTMLCanvasElement | null = null;

/** Mean luminance (0..1) of a 32x18 thumbnail of the current video frame, or null if unavailable. */
export function meanLuminance(video: HTMLVideoElement): number | null {
  try {
    canvas ??= document.createElement("canvas");
    canvas.width = 32;
    canvas.height = 18;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx || video.readyState < 2) return null;
    ctx.drawImage(video, 0, 0, 32, 18);
    const d = ctx.getImageData(0, 0, 32, 18).data;
    let s = 0;
    for (let i = 0; i < d.length; i += 4) s += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    return s / (d.length / 4) / 255;
  } catch {
    return null;
  }
}
