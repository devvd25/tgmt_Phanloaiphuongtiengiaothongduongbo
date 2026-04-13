const DEFAULT_IMAGE_TOPK = 4;
const DEFAULT_VIDEO_TOPK = 2;
const DEFAULT_VIDEO_MAX_FRAMES = 0;
const DEFAULT_LIVE_TOPK = 2;
const DEFAULT_LIVE_MAX_FRAMES = 1200;

const uiScaleSelect = document.getElementById("uiScaleSelect");

const imageInput = document.getElementById("imageInput");
const videoInput = document.getElementById("videoInput");

const btnChooseImage = document.getElementById("btnChooseImage");
const btnChooseVideo = document.getElementById("btnChooseVideo");
const btnOpenLive = document.getElementById("btnOpenLive");
const btnPredict = document.getElementById("btnPredict");
const btnStop = document.getElementById("btnStop");
const btnClear = document.getElementById("btnClear");
const btnSave = document.getElementById("btnSave");
const btnReplay = document.getElementById("btnReplay");
const btnZoom = document.getElementById("btnZoom");

const statusBar = document.getElementById("statusBar");

const inputPreview = document.getElementById("inputPreview");
const inputPlaceholder = document.getElementById("inputPlaceholder");
const inputName = document.getElementById("inputName");

const resultMain = document.getElementById("resultMain");
const resultSubTitle = document.getElementById("resultSubTitle");
const resultLines = document.getElementById("resultLines");
const resultMediaWrap = document.getElementById("resultMediaWrap");
const btnPlaySlow = document.getElementById("btnPlaySlow");
const btnPlayFast = document.getElementById("btnPlayFast");
const resultVideoPlayer = document.getElementById("resultVideoPlayer");
const resultImageView = document.getElementById("resultImageView");

const liveModal = document.getElementById("liveModal");
const btnCloseLiveModal = document.getElementById("btnCloseLiveModal");
const liveUrlInput = document.getElementById("liveUrlInput");
const savedLinksSelect = document.getElementById("savedLinksSelect");
const resolvedLiveOutput = document.getElementById("resolvedLiveOutput");
const btnResolveLive = document.getElementById("btnResolveLive");
const btnCopyResolvedLive = document.getElementById("btnCopyResolvedLive");
const btnLoadSavedLink = document.getElementById("btnLoadSavedLink");
const btnSaveLink = document.getElementById("btnSaveLink");
const btnDeleteLink = document.getElementById("btnDeleteLink");
const btnRunLive = document.getElementById("btnRunLive");

const zoomModal = document.getElementById("zoomModal");
const btnCloseZoomModal = document.getElementById("btnCloseZoomModal");
const btnZoomOut = document.getElementById("btnZoomOut");
const btnZoomIn = document.getElementById("btnZoomIn");
const btnZoomReset = document.getElementById("btnZoomReset");
const btnZoomFit = document.getElementById("btnZoomFit");
const zoomPercent = document.getElementById("zoomPercent");
const zoomViewport = document.getElementById("zoomViewport");
const zoomImage = document.getElementById("zoomImage");

const traceModal = document.getElementById("traceModal");
const btnCloseTraceModal = document.getElementById("btnCloseTraceModal");
const traceModalTitle = document.getElementById("traceModalTitle");
const traceSummary = document.getElementById("traceSummary");
const traceTabs = document.getElementById("traceTabs");
const traceStepTitle = document.getElementById("traceStepTitle");
const traceStepDesc = document.getElementById("traceStepDesc");
const traceStepImage = document.getElementById("traceStepImage");
const traceStepNoImage = document.getElementById("traceStepNoImage");

const state = {
  selectedKind: null,
  selectedFile: null,
  selectedLiveUrl: "",
  selectedSourceName: "",
  liveResolvedFor: "",
  liveResolvedSource: "",
  liveResolvedName: "",
  liveResolvedPredictUrl: "",
  liveStreamActive: false,
  liveStreamPreviewUrl: "",

  inputObjectUrl: "",
  previewType: "none",

  running: false,
  abortController: null,
  videoStreamSessionId: "",

  currentResult: null,
  currentFastVideoUrl: "",
  currentSlowVideoUrl: "",
  currentActiveVideoUrl: "",
  trace: null,
  traceStepIndex: 0,

  zoomScale: 1,
};

function setStatus(message, type = "info", clickable = false) {
  statusBar.className = `status ${type}`;
  statusBar.classList.toggle("clickable", Boolean(clickable));
  statusBar.textContent = clickable ? `${message} (Bấm vào đây để xem quy trình xử lý)` : message;
}

function looksLikeUrl(value) {
  const lower = String(value || "").trim().toLowerCase();
  return lower.startsWith("http://") || lower.startsWith("https://") || lower.startsWith("rtsp://") || lower.startsWith("rtmp://") || lower.startsWith("mms://");
}

function toRelativeUrl(url) {
  const raw = String(url || "").trim();
  if (!raw) return "";

  try {
    const parsed = new URL(raw, window.location.origin);
    return `${parsed.pathname}${parsed.search || ""}`;
  } catch (err) {
    return raw;
  }
}

function withCacheBust(url) {
  const clean = String(url || "").trim();
  if (!clean) return "";
  const sep = clean.includes("?") ? "&" : "?";
  return `${clean}${sep}_v=${Date.now()}`;
}

function revokeInputObjectUrl() {
  if (state.inputObjectUrl) {
    URL.revokeObjectURL(state.inputObjectUrl);
    state.inputObjectUrl = "";
  }
}

function clearInputPreview() {
  revokeInputObjectUrl();
  inputPreview.innerHTML = "";

  const p = document.createElement("p");
  p.id = "inputPlaceholder";
  p.textContent = "Chưa có ảnh";
  inputPreview.appendChild(p);

  inputName.textContent = "Chọn ảnh hoặc video để bắt đầu.";
  state.previewType = "none";
}

function renderInputImage(src, label = "", usesObjectUrl = false) {
  if (!usesObjectUrl) {
    revokeInputObjectUrl();
  }

  inputPreview.innerHTML = "";
  const img = document.createElement("img");
  img.className = "preview-image";
  img.alt = "Ảnh đầu vào";
  img.src = usesObjectUrl ? src : withCacheBust(src);
  inputPreview.appendChild(img);

  inputName.textContent = label || "Đã nạp ảnh.";
  state.previewType = "image";
}

function renderInputVideo(src, label = "", usesObjectUrl = false, autoPlay = false) {
  if (!usesObjectUrl) {
    revokeInputObjectUrl();
  }

  inputPreview.innerHTML = "";
  const video = document.createElement("video");
  video.className = "preview-video";
  video.controls = true;
  video.preload = "metadata";
  video.muted = true;
  video.playsInline = true;
  video.src = usesObjectUrl ? src : withCacheBust(src);
  inputPreview.appendChild(video);

  if (autoPlay) {
    video.currentTime = 0;
    video.play().catch(() => {});
  }

  inputName.textContent = label || "Đã nạp video.";
  state.previewType = "video";
}

function clearResultPanel() {
  resultMain.textContent = "OUTPUT: loại phương tiện";
  resultSubTitle.textContent = "Danh sách phương tiện phát hiện:";
  resultLines.textContent = "-";

  resultMediaWrap.classList.add("hidden");
  resultVideoPlayer.classList.add("hidden");
  resultImageView.classList.add("hidden");
  btnPlayFast.classList.add("hidden");
  btnPlaySlow.classList.add("hidden");

  resultVideoPlayer.pause();
  resultVideoPlayer.removeAttribute("src");
  resultVideoPlayer.load();
  resultImageView.removeAttribute("src");
}

function resetResultState() {
  state.currentResult = null;
  state.currentFastVideoUrl = "";
  state.currentSlowVideoUrl = "";
  state.currentActiveVideoUrl = "";
  state.trace = null;
  state.traceStepIndex = 0;
  clearResultPanel();
}

function resetSelectionState() {
  state.selectedKind = null;
  state.selectedFile = null;
  state.selectedLiveUrl = "";
  state.selectedSourceName = "";
  state.liveResolvedFor = "";
  state.liveResolvedSource = "";
  state.liveResolvedName = "";
  state.liveResolvedPredictUrl = "";
  state.liveStreamActive = false;
  state.liveStreamPreviewUrl = "";
  state.videoStreamSessionId = "";
}

function clearVideoStreamSessionState() {
  state.videoStreamSessionId = "";
}

function updateActionButtons() {
  const hasSource = Boolean(state.selectedKind);
  const hasResult = Boolean(state.currentResult);
  const hasReplay = Boolean(state.currentResult && state.currentResult.kind === "video" && (state.currentFastVideoUrl || state.currentSlowVideoUrl));
  const hasZoom = state.previewType === "image";
  const canZoomWhileRunning = state.liveStreamActive;

  btnPredict.disabled = state.running || !hasSource;
  btnChooseImage.disabled = state.running;
  btnChooseVideo.disabled = state.running;
  btnOpenLive.disabled = state.running;
  btnClear.disabled = state.running;

  btnStop.disabled = !state.running;
  btnSave.disabled = state.running || !hasResult;
  btnReplay.disabled = state.running || !hasReplay;
  btnZoom.disabled = !hasZoom || (state.running && !canZoomWhileRunning);

  btnSaveLink.disabled = state.running;
  btnDeleteLink.disabled = state.running;
  btnLoadSavedLink.disabled = state.running;
  btnResolveLive.disabled = state.running;
  btnCopyResolvedLive.disabled = state.running || !state.liveResolvedPredictUrl;
  btnRunLive.disabled = state.running;
}

function setRunning(isRunning) {
  state.running = Boolean(isRunning);
  btnPredict.textContent = state.running ? "Đang dự đoán..." : "Dự đoán";
  updateActionButtons();
}

function selectImageFile(file) {
  if (!file) return;

  stopLiveStreamPreview();
  clearVideoStreamSessionState();
  resetResultState();

  revokeInputObjectUrl();
  const objectUrl = URL.createObjectURL(file);
  state.inputObjectUrl = objectUrl;

  state.selectedKind = "image";
  state.selectedFile = file;
  state.selectedLiveUrl = "";
  state.selectedSourceName = file.name;

  renderInputImage(objectUrl, `Đã chọn: ${file.name}`, true);
  setStatus(`Đã chọn: ${file.name}`, "info");
  updateActionButtons();
}

function selectVideoFile(file) {
  if (!file) return;

  stopLiveStreamPreview();
  clearVideoStreamSessionState();
  resetResultState();

  revokeInputObjectUrl();
  const objectUrl = URL.createObjectURL(file);
  state.inputObjectUrl = objectUrl;

  state.selectedKind = "video";
  state.selectedFile = file;
  state.selectedLiveUrl = "";
  state.selectedSourceName = file.name;

  renderInputVideo(objectUrl, `Đã chọn video: ${file.name}`, true);
  setStatus(`Đã chọn video: ${file.name}`, "info");
  updateActionButtons();
}

function chooseLiveSource(liveUrl) {
  const normalized = String(liveUrl || "").trim();
  if (!normalized) return;

  stopLiveStreamPreview();
  clearVideoStreamSessionState();
  resetResultState();

  state.selectedKind = "live";
  state.selectedFile = null;
  state.selectedLiveUrl = normalized;
  state.selectedSourceName = normalized;
  if (state.liveResolvedFor !== normalized) {
    state.liveResolvedFor = "";
    state.liveResolvedSource = "";
    state.liveResolvedName = "";
    clearResolvedLiveOutput();
  }

  inputName.textContent = `Nguồn trực tiếp: ${normalized}`;
  setStatus("Đã chọn nguồn trực tiếp. Bấm Dự đoán để chạy.", "info");
  updateActionButtons();
}

function getLiveCandidateUrlFromModal() {
  return String(liveUrlInput.value || "").trim() || String(savedLinksSelect.value || "").trim();
}

function buildLivePredictStreamUrl(liveUrl, streamSource, streamName) {
  const params = new URLSearchParams();
  params.set("live_url", String(liveUrl || "").trim());
  params.set("topk", String(DEFAULT_LIVE_TOPK));
  params.set("max_frames", "0");
  params.set("_t", String(Date.now()));

  if (looksLikeUrl(streamSource)) {
    params.set("resolved_for_url", String(liveUrl || "").trim());
    params.set("stream_source", String(streamSource || "").trim());
    if (String(streamName || "").trim()) {
      params.set("stream_name", String(streamName || "").trim());
    }
  }

  return `${window.location.origin}/api/live/predict-stream?${params.toString()}`;
}

function clearResolvedLiveOutput() {
  state.liveResolvedPredictUrl = "";
  if (resolvedLiveOutput) {
    resolvedLiveOutput.value = "";
  }
}

function setResolvedLiveOutput(liveUrl, streamSource, streamName) {
  const safeLive = String(liveUrl || "").trim();
  const safeSource = String(streamSource || "").trim();
  const safeName = String(streamName || safeLive).trim();

  if (!safeLive || !safeSource) {
    clearResolvedLiveOutput();
    updateActionButtons();
    return;
  }

  const predictUrl = buildLivePredictStreamUrl(safeLive, safeSource, safeName);
  state.liveResolvedPredictUrl = predictUrl;

  if (resolvedLiveOutput) {
    resolvedLiveOutput.value = [
      `Nguồn gốc: ${safeLive}`,
      `Tên luồng: ${safeName}`,
      "",
      "Stream trực tiếp:",
      safeSource,
      "",
      "Predict-stream URL:",
      predictUrl,
    ].join("\n");
  }

  updateActionButtons();
}

async function resolveLiveUrlToStream(liveUrl, forceRefresh = false) {
  const normalized = String(liveUrl || "").trim();
  if (!normalized) {
    throw new Error("Thiếu link trực tiếp để đổi sang stream.");
  }

  const canReuse =
    !forceRefresh &&
    state.liveResolvedFor === normalized &&
    !!state.liveResolvedSource;

  if (canReuse) {
    return {
      streamSource: state.liveResolvedSource,
      streamName: state.liveResolvedName || normalized,
    };
  }

  const data = await apiRequest("/api/live/resolve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ live_url: normalized }),
  });

  const streamSource = String(data.stream_source || "").trim();
  const streamName = String(data.stream_name || normalized).trim();
  if (!streamSource) {
    throw new Error("Không lấy được luồng stream trực tiếp từ link YouTube.");
  }

  state.liveResolvedFor = normalized;
  state.liveResolvedSource = streamSource;
  state.liveResolvedName = streamName;

  return { streamSource, streamName };
}

function getInputPreviewVideoElement() {
  const video = inputPreview.querySelector("video.preview-video");
  return video instanceof HTMLVideoElement ? video : null;
}

async function resolveLivePreviewSource(forceRefresh = false) {
  const liveUrl = String(state.selectedLiveUrl || "").trim();
  if (!liveUrl) {
    throw new Error("Thiếu link trực tiếp để mở preview.");
  }
  return resolveLiveUrlToStream(liveUrl, forceRefresh);
}

async function openProcessingPreviewIfNeeded() {
  if (state.selectedKind === "video") {
    const previewVideo = getInputPreviewVideoElement();
    if (previewVideo) {
      previewVideo.currentTime = 0;
      previewVideo.play().catch(() => {});
      return;
    }

    if (state.inputObjectUrl) {
      renderInputVideo(
        state.inputObjectUrl,
        `Đang xử lý video: ${state.selectedSourceName || "video"}`,
        true,
        true
      );
    }
  }
}

async function pollVideoPredictionResult(sessionId) {
  if (!sessionId) return;
  if (state.videoStreamSessionId !== sessionId) return;

  try {
    const data = await apiRequest(`/api/video/stream-result?session_id=${encodeURIComponent(sessionId)}`);
    const streamStatus = String(data.stream_status || "").trim().toLowerCase();

    if ((streamStatus === "done" || streamStatus === "stopped") && data.result) {
      if (state.videoStreamSessionId !== sessionId) return;

      stopLiveStreamPreview();
      await renderResult(data.result);
      setStatus(data.result.main_text || "Dự đoán video thành công.", "ok", Boolean(state.trace));

      clearVideoStreamSessionState();
      setRunning(false);
      return;
    }

    if (streamStatus === "error") {
      throw new Error(String(data.error || "Lỗi xử lý video realtime."));
    }
  } catch (err) {
    if (state.videoStreamSessionId !== sessionId) return;
    stopLiveStreamPreview();
    clearVideoStreamSessionState();
    setRunning(false);
    setStatus((err && err.message) || "Không lấy được kết quả video realtime.", "error");
    return;
  }

  if (!state.running) return;
  if (state.videoStreamSessionId !== sessionId) return;

  setTimeout(() => {
    pollVideoPredictionResult(sessionId);
  }, 700);
}

async function startVideoPredictionRealtime() {
  if (!state.selectedFile) {
    throw new Error("Hãy chọn video trước khi dự đoán.");
  }

  clearVideoStreamSessionState();

  const formData = new FormData();
  formData.append("file", state.selectedFile);
  formData.append("topk", String(DEFAULT_VIDEO_TOPK));
  formData.append("max_frames", String(DEFAULT_VIDEO_MAX_FRAMES));

  const data = await apiRequest("/api/video/prepare-stream", {
    method: "POST",
    body: formData,
  });

  const sessionId = String(data.session_id || "").trim();
  const streamUrl = toRelativeUrl(data.stream_url || "");
  const sourceName = String(data.source_name || state.selectedSourceName || "video").trim();

  if (!sessionId || !streamUrl) {
    throw new Error("Không khởi tạo được luồng dự đoán video realtime.");
  }

  state.videoStreamSessionId = sessionId;

  const guideTrace = buildRealtimeGuideTrace("video", sourceName);
  state.trace = guideTrace;
  state.traceStepIndex = 0;

  startLiveStreamPreview(streamUrl, `Đang dự đoán video: ${sourceName}`);

  resultMain.textContent = "OUTPUT: dự đoán video theo thời gian thực";
  resultSubTitle.textContent = "Tóm tắt kết quả video:";
  setResultLines(guideTrace.summary_lines);

  setStatus("Đang dự đoán video realtime.", "ok", Boolean(state.trace));
  pollVideoPredictionResult(sessionId);
}

async function startLivePredictionRealtime() {
  const liveUrl = String(state.selectedLiveUrl || "").trim();
  if (!liveUrl) {
    throw new Error("Hãy nhập link trực tiếp trước khi dự đoán.");
  }

  let streamName = liveUrl;
  try {
    const resolved = await resolveLivePreviewSource(false);
    streamName = resolved.streamName || liveUrl;
    setResolvedLiveOutput(liveUrl, resolved.streamSource, streamName);
  } catch (err) {
    // Continue even if resolve metadata fails; stream route will retry resolve.
    clearResolvedLiveOutput();
    updateActionButtons();
  }

  const streamParams = new URLSearchParams();
  streamParams.set("live_url", liveUrl);
  streamParams.set("topk", String(DEFAULT_LIVE_TOPK));
  streamParams.set("max_frames", "0");
  streamParams.set("_t", String(Date.now()));

  if (
    state.liveResolvedFor === liveUrl &&
    state.liveResolvedSource &&
    looksLikeUrl(state.liveResolvedSource)
  ) {
    streamParams.set("resolved_for_url", liveUrl);
    streamParams.set("stream_source", state.liveResolvedSource);
    streamParams.set("stream_name", state.liveResolvedName || streamName);
  }

  const streamUrl = `/api/live/predict-stream?${streamParams.toString()}`;

  const guideTrace = buildRealtimeGuideTrace("live", streamName);
  state.trace = guideTrace;
  state.traceStepIndex = 0;

  startLiveStreamPreview(
    streamUrl,
    `Đang phát dự đoán trực tiếp: ${streamName}`
  );

  resultMain.textContent = "OUTPUT: dự đoán trực tiếp theo thời gian thực";
  resultSubTitle.textContent = "Tóm tắt trực tiếp:";
  setResultLines(guideTrace.summary_lines);

  setStatus(
    "Đang dự đoán trực tiếp realtime. Bấm Dừng video để dừng.",
    "ok",
    Boolean(state.trace)
  );
}

async function apiRequest(url, options = {}) {
  const response = await fetch(url, options);
  let payload = null;

  try {
    payload = await response.json();
  } catch (err) {
    throw new Error("Máy chủ trả về dữ liệu không hợp lệ.");
  }

  if (!response.ok || payload.status !== "ok") {
    const message = (payload && payload.message) || "Đã xảy ra lỗi không xác định.";
    throw new Error(message);
  }

  return payload;
}

async function refreshSavedLinks() {
  const data = await apiRequest("/api/live-links");
  renderSavedLinks(data.links || []);
}

function renderSavedLinks(links) {
  savedLinksSelect.innerHTML = "";

  const safeLinks = Array.isArray(links) ? links : [];
  if (safeLinks.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "(Chưa có link)";
    savedLinksSelect.appendChild(option);
    return;
  }

  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = "(Không dùng link đã lưu)";
  savedLinksSelect.appendChild(blank);

  for (const link of safeLinks) {
    const option = document.createElement("option");
    option.value = link;
    option.textContent = link;
    savedLinksSelect.appendChild(option);
  }
}

function openModal(modalEl) {
  modalEl.classList.remove("hidden");
}

function closeModal(modalEl) {
  modalEl.classList.add("hidden");
}

function stopLiveStreamPreview() {
  if (!state.liveStreamActive) return;

  const previewImg = inputPreview.querySelector("img.preview-image");
  if (previewImg) {
    previewImg.removeAttribute("src");
  }

  state.liveStreamActive = false;
  state.liveStreamPreviewUrl = "";
}

function startLiveStreamPreview(previewUrl, labelText) {
  const safePreview = String(previewUrl || "").trim();
  if (!safePreview) return;

  stopLiveStreamPreview();
  state.liveStreamPreviewUrl = safePreview;
  state.liveStreamActive = true;
  renderInputImage(safePreview, labelText || "Đang phát dự đoán trực tiếp.", false);
}

async function ensureWebPlayableVideo(url) {
  const relative = toRelativeUrl(url);
  if (!relative) return "";
  if (!relative.startsWith("/web_outputs/")) return relative;
  if (relative.toLowerCase().endsWith("_web.mp4")) return relative;

  try {
    const data = await apiRequest("/api/video/ensure-web", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_url: relative }),
    });
    return toRelativeUrl(data.video_url || relative);
  } catch (err) {
    return relative;
  }
}

async function runImagePrediction(signal) {
  if (!state.selectedFile) {
    throw new Error("Hãy chọn ảnh trước khi dự đoán.");
  }

  const formData = new FormData();
  formData.append("file", state.selectedFile);
  formData.append("topk", String(DEFAULT_IMAGE_TOPK));

  const data = await apiRequest("/api/predict/image", {
    method: "POST",
    body: formData,
    signal,
  });

  return data.result;
}

async function runVideoPrediction(signal) {
  if (!state.selectedFile) {
    throw new Error("Hãy chọn video trước khi dự đoán.");
  }

  const formData = new FormData();
  formData.append("file", state.selectedFile);
  formData.append("topk", String(DEFAULT_VIDEO_TOPK));
  formData.append("max_frames", String(DEFAULT_VIDEO_MAX_FRAMES));

  const data = await apiRequest("/api/predict/video", {
    method: "POST",
    body: formData,
    signal,
  });

  return data.result;
}

async function runLivePrediction(signal) {
  const liveUrl = String(state.selectedLiveUrl || "").trim();
  if (!liveUrl) {
    throw new Error("Hãy nhập link trực tiếp trước khi dự đoán.");
  }

  const payload = {
    live_url: liveUrl,
    topk: DEFAULT_LIVE_TOPK,
    max_frames: DEFAULT_LIVE_MAX_FRAMES,
  };

  if (
    state.liveResolvedFor === liveUrl &&
    state.liveResolvedSource &&
    looksLikeUrl(state.liveResolvedSource)
  ) {
    payload.stream_source = state.liveResolvedSource;
    payload.stream_name = state.liveResolvedName || liveUrl;
    payload.resolved_for_url = liveUrl;
  }

  const data = await apiRequest("/api/predict/live", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  if (Array.isArray(data.links)) {
    renderSavedLinks(data.links);
  }

  return data.result;
}

function setResultLines(lines) {
  if (!Array.isArray(lines) || lines.length === 0) {
    resultLines.textContent = "-";
    return;
  }

  resultLines.textContent = lines.map((line) => `- ${line}`).join("\n");
}

function setResultVideoSource(url, autoPlay = false) {
  const source = toRelativeUrl(url);
  if (!source) return;

  state.currentActiveVideoUrl = source;

  resultMediaWrap.classList.remove("hidden");
  resultImageView.classList.add("hidden");
  resultVideoPlayer.classList.remove("hidden");

  resultVideoPlayer.src = withCacheBust(source);
  resultVideoPlayer.load();

  if (autoPlay) {
    resultVideoPlayer.play().catch(() => {});
  }
}

function showImageResultMedia(imageUrl) {
  const src = toRelativeUrl(imageUrl);
  if (!src) {
    resultMediaWrap.classList.add("hidden");
    return;
  }

  resultMediaWrap.classList.remove("hidden");
  resultVideoPlayer.classList.add("hidden");
  resultImageView.classList.remove("hidden");
  btnPlayFast.classList.add("hidden");
  btnPlaySlow.classList.add("hidden");
  resultImageView.src = withCacheBust(src);
}

async function showVideoResultMedia(result) {
  const fast = toRelativeUrl(result.fast_video_url);
  const slow = toRelativeUrl(result.slow_video_url);

  state.currentFastVideoUrl = fast;
  state.currentSlowVideoUrl = slow;

  btnPlayFast.classList.toggle("hidden", !fast);
  btnPlaySlow.classList.toggle("hidden", !slow);

  if (fast) {
    btnPlayFast.textContent = "Xem bản nhanh";
  }
  if (slow) {
    btnPlaySlow.textContent = "Xem bản chậm";
  }

  const initial = fast || slow;
  if (!initial) {
    resultMediaWrap.classList.add("hidden");
    return;
  }

  setResultVideoSource(initial, true);

  // Optimize only the active source in background so the result appears immediately.
  optimizeVideoSourceInBackground(initial);
}

function normalizeVideoSourceForState(rawSource, optimizedSource) {
  const raw = toRelativeUrl(rawSource);
  const optimized = toRelativeUrl(optimizedSource) || raw;

  if (raw && state.currentFastVideoUrl === raw) {
    state.currentFastVideoUrl = optimized;
  }
  if (raw && state.currentSlowVideoUrl === raw) {
    state.currentSlowVideoUrl = optimized;
  }
  if (raw && state.currentActiveVideoUrl === raw) {
    state.currentActiveVideoUrl = optimized;
  }

  return optimized;
}

async function ensureAndPlayVideoSource(rawSource) {
  const source = toRelativeUrl(rawSource);
  if (!source) return;

  const optimized = await ensureWebPlayableVideo(source);
  const playable = normalizeVideoSourceForState(source, optimized || source);
  setResultVideoSource(playable, true);
}

async function optimizeVideoSourceInBackground(rawSource) {
  const source = toRelativeUrl(rawSource);
  if (!source) return;

  try {
    const isActiveNow = state.currentActiveVideoUrl === source;
    const optimized = await ensureWebPlayableVideo(source);
    const playable = normalizeVideoSourceForState(source, optimized || source);
    if (isActiveNow && playable !== source) {
      setResultVideoSource(playable, false);
    }
  } catch (err) {
    // Ignore optimization errors; playback can still fallback to original source.
  }
}

function getCurrentPreviewImageUrl() {
  const img = inputPreview.querySelector("img.preview-image");
  return img ? img.src : "";
}

function buildRealtimeGuideTrace(mode, sourceName) {
  const isLive = mode === "live";
  const safeSource = String(sourceName || "").trim() || (isLive ? "Nguồn live" : "Video tải lên");
  const modeLabel = isLive ? "trực tiếp (LIVE)" : "video realtime";

  const summaryLines = [
    `A. Chế độ: ${modeLabel}.`,
    `B. Nguồn dữ liệu: ${safeSource}.`,
    "C. Mỗi frame được resize (giới hạn cạnh dài) để giữ tốc độ xử lý.",
    "D. YOLO phát hiện box phương tiện theo từng frame.",
    "E. Lọc nhiễu realtime: bỏ box yếu/sát biên và siết ngưỡng với Truck.",
    "F. Nếu không có box hợp lệ, hệ thống fallback bằng multi-crop + VGG16.",
    "G. Vẽ nhãn tiếng Việt + confidence trực tiếp lên frame.",
    isLive
      ? "H. Luồng live hiển thị liên tục; bấm Dừng video để dừng ngay."
      : "H. Luồng video mở ngay, đồng thời ghi replay để trả kết quả tổng hợp.",
    "I. Có thể bấm vào thanh trạng thái để xem chi tiết quy trình từng bước.",
  ];

  const steps = [
    {
      title: "A. Khởi tạo nguồn đầu vào",
      description: `Nhận nguồn ${isLive ? "live" : "video"}: ${safeSource}.`,
      image_url: "",
    },
    {
      title: "B. Tiền xử lý theo frame",
      description: "Frame được resize về kích thước phù hợp để cân bằng tốc độ và độ chính xác realtime.",
      image_url: "",
    },
    {
      title: "C. Detector YOLO",
      description: "YOLO quét frame, sinh box và confidence cho các phương tiện.",
      image_url: "",
    },
    {
      title: "D. Bộ lọc nhiễu realtime",
      description: "Loại các box dễ nhiễu trong bối cảnh video/live để giảm dự đoán sai.",
      image_url: "",
    },
    {
      title: "E. Nhánh fallback",
      description: "Khi detector không đủ tin cậy, frame sẽ được multi-crop và phân loại bằng VGG16.",
      image_url: "",
    },
    {
      title: "F. Hậu xử lý hiển thị",
      description: "Vẽ box, nhãn tiếng Việt, confidence và thông tin frame ngay trên luồng hình.",
      image_url: "",
    },
    {
      title: "Z. Kết thúc phiên",
      description: isLive
        ? "Người dùng bấm Dừng video để kết thúc luồng live realtime."
        : "Khi xử lý xong hoặc bấm dừng, hệ thống trả kết quả tổng hợp và video replay.",
      image_url: "",
    },
  ];

  return {
    title: `Quy trình xử lý ${modeLabel} (A-Z)`,
    summary_lines: summaryLines,
    steps,
  };
}

function openTraceModalFromState() {
  const trace = state.trace;
  if (!trace || !Array.isArray(trace.steps) || trace.steps.length === 0) {
    setStatus("Chưa có dữ liệu quy trình xử lý để hiển thị.", "warn");
    return;
  }

  traceModalTitle.textContent = trace.title || "Quy trình xử lý";

  const summaryLines = Array.isArray(trace.summary_lines) ? trace.summary_lines : [];
  if (summaryLines.length > 0) {
    const ul = document.createElement("ul");
    for (const line of summaryLines) {
      const li = document.createElement("li");
      li.textContent = line;
      ul.appendChild(li);
    }
    traceSummary.innerHTML = "";
    traceSummary.appendChild(ul);
  } else {
    traceSummary.textContent = "Không có tóm tắt.";
  }

  traceTabs.innerHTML = "";
  trace.steps.forEach((step, idx) => {
    const tabBtn = document.createElement("button");
    tabBtn.type = "button";
    tabBtn.className = "btn-secondary trace-tab";
    tabBtn.textContent = step.title || `Bước ${idx + 1}`;
    tabBtn.addEventListener("click", () => selectTraceStep(idx));
    traceTabs.appendChild(tabBtn);
  });

  state.traceStepIndex = 0;
  selectTraceStep(0);
  openModal(traceModal);
}

function selectTraceStep(index) {
  const trace = state.trace;
  if (!trace || !Array.isArray(trace.steps) || trace.steps.length === 0) return;

  const nextIndex = Math.max(0, Math.min(trace.steps.length - 1, index));
  state.traceStepIndex = nextIndex;

  const step = trace.steps[nextIndex] || {};
  traceStepTitle.textContent = step.title || `Bước ${nextIndex + 1}`;
  traceStepDesc.textContent = step.description || "";

  const imageUrl = toRelativeUrl(step.image_url);
  if (imageUrl) {
    traceStepImage.classList.remove("hidden");
    traceStepNoImage.classList.add("hidden");
    traceStepImage.src = withCacheBust(imageUrl);
  } else {
    traceStepImage.classList.add("hidden");
    traceStepNoImage.classList.remove("hidden");
    traceStepImage.removeAttribute("src");
  }

  const tabs = traceTabs.querySelectorAll(".trace-tab");
  tabs.forEach((tab, idx) => {
    tab.classList.toggle("active", idx === nextIndex);
  });
}

function clampZoom(scale) {
  return Math.max(0.2, Math.min(6.0, scale));
}

function applyZoom() {
  zoomImage.style.transform = `scale(${state.zoomScale})`;
  zoomPercent.textContent = `${Math.round(state.zoomScale * 100)}%`;
}

function centerZoomViewport() {
  requestAnimationFrame(() => {
    const maxLeft = Math.max(0, zoomViewport.scrollWidth - zoomViewport.clientWidth);
    zoomViewport.scrollLeft = Math.round(maxLeft / 2);
  });
}

function zoomIn() {
  state.zoomScale = clampZoom(state.zoomScale * 1.15);
  applyZoom();
}

function zoomOut() {
  state.zoomScale = clampZoom(state.zoomScale / 1.15);
  applyZoom();
}

function zoomReset() {
  state.zoomScale = 1;
  applyZoom();
  centerZoomViewport();
}

function zoomFit() {
  if (!zoomImage.naturalWidth || !zoomImage.naturalHeight) {
    state.zoomScale = 1;
    applyZoom();
    centerZoomViewport();
    return;
  }

  const vw = Math.max(1, zoomViewport.clientWidth - 16);
  const vh = Math.max(1, zoomViewport.clientHeight - 16);
  const ratio = Math.min(vw / zoomImage.naturalWidth, vh / zoomImage.naturalHeight);

  state.zoomScale = clampZoom(ratio);
  applyZoom();
  centerZoomViewport();
}

function openZoomModalFromPreview() {
  const src = getCurrentPreviewImageUrl();
  if (!src) {
    setStatus("Chưa có ảnh để phóng to.", "warn");
    return;
  }

  zoomImage.onload = () => {
    zoomFit();
    zoomViewport.scrollTop = 0;
    centerZoomViewport();
  };
  zoomImage.src = src;
  openModal(zoomModal);
}

function triggerDownload(url, fileName = "") {
  const safeUrl = toRelativeUrl(url);
  if (!safeUrl) return;

  const anchor = document.createElement("a");
  anchor.href = safeUrl;
  anchor.download = fileName || "";
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

function getDownloadNameFromUrl(url, fallback) {
  const relative = toRelativeUrl(url);
  if (!relative) return fallback;

  const last = relative.split("/").pop() || "";
  if (!last) return fallback;
  return last.split("?")[0] || fallback;
}

async function renderResult(result) {
  state.currentResult = result;
  state.trace = result.trace || null;

  const summaryLines = Array.isArray(result.summary_lines) ? result.summary_lines : [];
  resultMain.textContent = result.main_text || "OUTPUT: loại phương tiện";

  if (result.kind === "image") {
    resultSubTitle.textContent = "Danh sách phương tiện phát hiện:";
    setResultLines(summaryLines);

    const imageUrl = toRelativeUrl(result.image_url);
    showImageResultMedia(imageUrl);
    if (imageUrl) {
      renderInputImage(imageUrl, "Ảnh kết quả sau khi dự đoán.", false);
    }

    state.currentFastVideoUrl = "";
    state.currentSlowVideoUrl = "";
    state.currentActiveVideoUrl = "";
  } else {
    resultSubTitle.textContent = result.is_live ? "Tóm tắt kết quả trực tiếp:" : "Tóm tắt kết quả video:";
    setResultLines(summaryLines);
    await showVideoResultMedia(result);

    if (result.preview_url) {
      renderInputImage(result.preview_url, "Khung hình cuối sau khi xử lý video.", false);
    }
  }

  updateActionButtons();
}

async function runPrediction() {
  if (state.running) return;

  if (!state.selectedKind) {
    setStatus("Hãy chọn ảnh, video hoặc link trực tiếp trước khi dự đoán.", "warn");
    return;
  }

  let keepRunningState = false;
  try {
    setRunning(true);

    if (state.selectedKind === "live") {
      setStatus("Đang mở luồng dự đoán trực tiếp...", "info");
      await startLivePredictionRealtime();
      keepRunningState = true;
      return;
    }

    if (state.selectedKind === "video") {
      setStatus("Đang mở video và chạy dự đoán realtime...", "info");
      await startVideoPredictionRealtime();
      keepRunningState = true;
      return;
    }

    setStatus("Đang xử lý, vui lòng chờ...", "info");

    state.abortController = new AbortController();
    const signal = state.abortController.signal;

    let result = null;
    if (state.selectedKind === "image") {
      result = await runImagePrediction(signal);
    } else if (state.selectedKind === "video") {
      result = await runVideoPrediction(signal);
    } else {
      result = await runLivePrediction(signal);
    }

    await renderResult(result);
    setStatus(result.main_text || "Dự đoán thành công.", "ok", Boolean(state.trace));
  } catch (err) {
    if (err && err.name === "AbortError") {
      setStatus("Đã dừng yêu cầu dự đoán trên giao diện web.", "warn");
    } else {
      setStatus((err && err.message) || "Đã xảy ra lỗi trong quá trình dự đoán.", "error");
    }
  } finally {
    state.abortController = null;
    if (!keepRunningState) {
      setRunning(false);
    } else {
      updateActionButtons();
    }
  }
}

function stopPrediction() {
  if (state.selectedKind === "video" && state.videoStreamSessionId) {
    const sessionId = state.videoStreamSessionId;

    setStatus("Đang dừng video...", "warn");
    stopLiveStreamPreview();
    clearVideoStreamSessionState();
    setRunning(false);

    apiRequest("/api/video/stop-stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    }).catch(() => {});
    return;
  }

  if (state.selectedKind === "live" && state.liveStreamActive) {
    stopLiveStreamPreview();
    setStatus("Đã dừng dự đoán trực tiếp.", "ok", Boolean(state.trace));
    setRunning(false);
    return;
  }

  if (!state.running || !state.abortController) return;
  setStatus("Đang dừng video...", "warn");
  state.abortController.abort();
}

function clearAll() {
  if (state.running) return;

  stopLiveStreamPreview();
  clearVideoStreamSessionState();
  resetSelectionState();
  resetResultState();
  clearInputPreview();

  imageInput.value = "";
  videoInput.value = "";

  setStatus("Đã xóa kết quả. Hãy chọn ảnh mới.", "info");
  updateActionButtons();
}

function onSaveResult() {
  if (!state.currentResult) {
    setStatus("Chưa có kết quả để lưu.", "warn");
    return;
  }

  if (state.currentResult.kind === "image") {
    const imageUrl = toRelativeUrl(state.currentResult.image_url);
    if (!imageUrl) {
      setStatus("Không tìm thấy ảnh kết quả để tải.", "error");
      return;
    }

    const fileName = getDownloadNameFromUrl(imageUrl, "ket_qua_anh.png");
    triggerDownload(imageUrl, fileName);
    setStatus("Đã gửi yêu cầu tải ảnh kết quả.", "ok", Boolean(state.trace));
    return;
  }

  const fast = toRelativeUrl(state.currentFastVideoUrl);
  const slow = toRelativeUrl(state.currentSlowVideoUrl);
  const toDownload = [fast, slow].filter(Boolean);

  if (toDownload.length === 0) {
    setStatus("Không có video kết quả để tải.", "warn", Boolean(state.trace));
    return;
  }

  toDownload.forEach((url, idx) => {
    const fileName = getDownloadNameFromUrl(url, idx === 0 ? "ket_qua_video_nhanh.mp4" : "ket_qua_video_cham.mp4");
    setTimeout(() => triggerDownload(url, fileName), idx * 180);
  });

  setStatus("Đã gửi yêu cầu tải video kết quả (nhanh/chậm).", "ok", Boolean(state.trace));
}

function onReplayResult() {
  if (!state.currentResult || state.currentResult.kind !== "video") {
    setStatus("Chưa có video kết quả để xem lại.", "warn");
    return;
  }

  const replay = state.currentFastVideoUrl || state.currentSlowVideoUrl;
  if (!replay) {
    setStatus("Không tìm thấy video kết quả để xem lại.", "error", Boolean(state.trace));
    return;
  }

  ensureAndPlayVideoSource(replay);
  setStatus("Đang phát lại video kết quả.", "ok", Boolean(state.trace));
}

async function onSaveLiveLink() {
  const candidate = String(liveUrlInput.value || "").trim() || String(savedLinksSelect.value || "").trim();
  if (!candidate) {
    setStatus("Hãy nhập hoặc chọn một link trước khi lưu.", "warn");
    return;
  }
  if (!looksLikeUrl(candidate)) {
    setStatus("Link không hợp lệ. Link phải bắt đầu bằng http:// hoặc https://", "warn");
    return;
  }

  try {
    const data = await apiRequest("/api/live-links", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: candidate }),
    });

    renderSavedLinks(data.links || []);
    savedLinksSelect.value = candidate;
    liveUrlInput.value = candidate;
    setStatus("Đã lưu link trực tiếp.", "ok");
  } catch (err) {
    setStatus((err && err.message) || "Không lưu được link trực tiếp.", "error");
  }
}

async function onDeleteLiveLink() {
  const target = String(savedLinksSelect.value || "").trim() || String(liveUrlInput.value || "").trim();
  if (!target) {
    setStatus("Chưa chọn link để xóa.", "warn");
    return;
  }

  try {
    const data = await apiRequest(`/api/live-links?url=${encodeURIComponent(target)}`, {
      method: "DELETE",
    });
    renderSavedLinks(data.links || []);
    liveUrlInput.value = "";

    if (target === state.liveResolvedFor) {
      state.liveResolvedFor = "";
      state.liveResolvedSource = "";
      state.liveResolvedName = "";
      clearResolvedLiveOutput();
    }

    setStatus("Đã xóa link trực tiếp đã lưu.", "ok");
  } catch (err) {
    setStatus((err && err.message) || "Không xóa được link trực tiếp.", "error");
  }
}

function onLoadSavedLink() {
  const selected = String(savedLinksSelect.value || "").trim();
  if (!selected) {
    setStatus("Chưa chọn link đã lưu để nạp.", "warn");
    return;
  }

  liveUrlInput.value = selected;

  if (selected === state.liveResolvedFor && state.liveResolvedSource) {
    setResolvedLiveOutput(selected, state.liveResolvedSource, state.liveResolvedName || selected);
  } else {
    clearResolvedLiveOutput();
    updateActionButtons();
  }
}

async function onResolveLiveLink() {
  const candidate = getLiveCandidateUrlFromModal();
  if (!candidate) {
    setStatus("Hãy nhập link YouTube trước khi đổi link.", "warn");
    return;
  }
  if (!looksLikeUrl(candidate)) {
    setStatus("Link không hợp lệ. Link phải bắt đầu bằng http:// hoặc https://", "warn");
    return;
  }

  try {
    setStatus("Đang đổi link YouTube sang stream trực tiếp...", "info");
    liveUrlInput.value = candidate;

    const resolved = await resolveLiveUrlToStream(candidate, true);
    setResolvedLiveOutput(candidate, resolved.streamSource, resolved.streamName);

    setStatus("Đã đổi link thành công. Bấm Dự đoán trực tiếp để chạy.", "ok");
  } catch (err) {
    clearResolvedLiveOutput();
    setStatus((err && err.message) || "Không đổi được link YouTube.", "error");
  }
}

async function onCopyResolvedLive() {
  const text = String(state.liveResolvedPredictUrl || "").trim();
  if (!text) {
    setStatus("Chưa có link đã đổi để copy.", "warn");
    return;
  }

  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const temp = document.createElement("textarea");
      temp.value = text;
      temp.setAttribute("readonly", "");
      temp.style.position = "absolute";
      temp.style.left = "-9999px";
      document.body.appendChild(temp);
      temp.select();
      document.execCommand("copy");
      temp.remove();
    }

    setStatus("Đã copy link predict-stream đã đổi.", "ok");
  } catch (err) {
    setStatus("Không copy được link đã đổi trên trình duyệt này.", "warn");
  }
}

function runLiveFromModal() {
  const candidate = String(liveUrlInput.value || "").trim() || String(savedLinksSelect.value || "").trim();
  if (!candidate) {
    setStatus("Hãy nhập link trực tiếp trước.", "warn");
    return;
  }
  if (!looksLikeUrl(candidate)) {
    setStatus("Link không hợp lệ. Link phải bắt đầu bằng http:// hoặc https://", "warn");
    return;
  }

  chooseLiveSource(candidate);
  closeModal(liveModal);
  runPrediction();
}

function onScaleChange() {
  const selected = uiScaleSelect.value || "auto";

  if (selected === "auto") {
    let scale = 1.0;
    const w = window.innerWidth;
    const h = window.innerHeight;

    if (w < 1200) {
      scale -= 0.08;
    } else if (w < 1400) {
      scale -= 0.03;
    } else if (w > 1780) {
      scale += 0.05;
    }

    if (h < 860) {
      scale -= 0.06;
    } else if (h > 1060) {
      scale += 0.03;
    }

    scale = Math.max(0.84, Math.min(1.12, scale));
    document.documentElement.style.setProperty("--ui-scale", String(scale));
    return;
  }

  const scale = Number(selected || "1");
  document.documentElement.style.setProperty("--ui-scale", String(scale));
}

function closeTopModalOnEscape(event) {
  if (event.key !== "Escape") return;

  if (!traceModal.classList.contains("hidden")) {
    closeModal(traceModal);
    return;
  }
  if (!zoomModal.classList.contains("hidden")) {
    closeModal(zoomModal);
    return;
  }
  if (!liveModal.classList.contains("hidden")) {
    closeModal(liveModal);
  }
}

function bindModalBackdropClose(modal) {
  modal.addEventListener("click", (event) => {
    if (event.target === modal) {
      closeModal(modal);
    }
  });
}

function wireEvents() {
  btnChooseImage.addEventListener("click", () => imageInput.click());
  btnChooseVideo.addEventListener("click", () => videoInput.click());
  btnOpenLive.addEventListener("click", () => {
    openModal(liveModal);

    const candidate = getLiveCandidateUrlFromModal();
    if (candidate && candidate === state.liveResolvedFor && state.liveResolvedSource) {
      setResolvedLiveOutput(candidate, state.liveResolvedSource, state.liveResolvedName || candidate);
    } else {
      clearResolvedLiveOutput();
      updateActionButtons();
    }
  });

  imageInput.addEventListener("change", () => {
    const file = imageInput.files && imageInput.files[0] ? imageInput.files[0] : null;
    if (file) {
      selectImageFile(file);
    }
  });

  videoInput.addEventListener("change", () => {
    const file = videoInput.files && videoInput.files[0] ? videoInput.files[0] : null;
    if (file) {
      selectVideoFile(file);
    }
  });

  btnPredict.addEventListener("click", runPrediction);
  btnStop.addEventListener("click", stopPrediction);
  btnClear.addEventListener("click", clearAll);
  btnSave.addEventListener("click", onSaveResult);
  btnReplay.addEventListener("click", onReplayResult);
  btnZoom.addEventListener("click", openZoomModalFromPreview);

  btnPlayFast.addEventListener("click", () => ensureAndPlayVideoSource(state.currentFastVideoUrl));
  btnPlaySlow.addEventListener("click", () => ensureAndPlayVideoSource(state.currentSlowVideoUrl));

  resultVideoPlayer.addEventListener("error", async () => {
    const currentSrc = toRelativeUrl(resultVideoPlayer.currentSrc || resultVideoPlayer.src);
    if (!currentSrc) return;

    const repaired = await ensureWebPlayableVideo(currentSrc);
    if (repaired && repaired !== currentSrc) {
      const playable = normalizeVideoSourceForState(currentSrc, repaired);
      setResultVideoSource(playable, true);
      setStatus("Đã tự chuyển sang bản video tương thích trình duyệt.", "warn", Boolean(state.trace));
    }
  });

  statusBar.addEventListener("click", () => {
    if (!statusBar.classList.contains("clickable")) return;
    openTraceModalFromState();
  });

  btnCloseLiveModal.addEventListener("click", () => closeModal(liveModal));
  btnResolveLive.addEventListener("click", onResolveLiveLink);
  btnCopyResolvedLive.addEventListener("click", onCopyResolvedLive);
  btnLoadSavedLink.addEventListener("click", onLoadSavedLink);
  btnSaveLink.addEventListener("click", onSaveLiveLink);
  btnDeleteLink.addEventListener("click", onDeleteLiveLink);
  btnRunLive.addEventListener("click", runLiveFromModal);

  liveUrlInput.addEventListener("input", () => {
    const current = String(liveUrlInput.value || "").trim();
    if (!current || current !== state.liveResolvedFor) {
      clearResolvedLiveOutput();
      updateActionButtons();
    }
  });

  savedLinksSelect.addEventListener("change", () => {
    if (savedLinksSelect.value) {
      liveUrlInput.value = savedLinksSelect.value;
    }

    const current = String(liveUrlInput.value || "").trim();
    if (current && current === state.liveResolvedFor && state.liveResolvedSource) {
      setResolvedLiveOutput(current, state.liveResolvedSource, state.liveResolvedName || current);
    } else {
      clearResolvedLiveOutput();
      updateActionButtons();
    }
  });

  btnCloseZoomModal.addEventListener("click", () => closeModal(zoomModal));
  btnZoomIn.addEventListener("click", zoomIn);
  btnZoomOut.addEventListener("click", zoomOut);
  btnZoomReset.addEventListener("click", zoomReset);
  btnZoomFit.addEventListener("click", zoomFit);

  btnCloseTraceModal.addEventListener("click", () => closeModal(traceModal));

  uiScaleSelect.addEventListener("change", onScaleChange);
  window.addEventListener("resize", () => {
    if ((uiScaleSelect.value || "") === "auto") {
      onScaleChange();
    }
  });

  document.addEventListener("keydown", closeTopModalOnEscape);

  bindModalBackdropClose(liveModal);
  bindModalBackdropClose(zoomModal);
  bindModalBackdropClose(traceModal);
}

async function bootstrap() {
  try {
    wireEvents();
    onScaleChange();

    resetSelectionState();
    resetResultState();
    clearInputPreview();

    await refreshSavedLinks();
    setStatus("Máy chủ đã sẵn sàng. Hãy chọn ảnh để dự đoán.", "ok");
  } catch (err) {
    setStatus((err && err.message) || "Không khởi tạo được giao diện.", "error");
  } finally {
    updateActionButtons();
  }
}

bootstrap();