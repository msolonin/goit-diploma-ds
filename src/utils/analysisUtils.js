export const THRESHOLD = 90;

export function parseAnalysisData() {
  const LOCAL_STORAGE_KEY = "boatFiles";
  const storedData = JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY) || "[]");

  return storedData.map((item) => {
    const photo = item.analysis?.photo_type;
    const modelType = item.analysis?.model_type;
    const modelName = item.analysis?.model_name;

    let title = "";
    let borderColor = "green";

    if (!photo || photo.percent < THRESHOLD) {
      title = "unrecognized photo type";
      borderColor = "red";
    } else if (photo.percent >= THRESHOLD && modelType && modelType.percent < THRESHOLD) {
      title = "unrecognized model type";
      borderColor = "red";
    } else {
      const parts = [];
      if (photo?.target) parts.push(`type: ${photo.target}`);
      if (modelType?.target) parts.push(`boat: ${modelType.target}`);
      if (modelName?.target) parts.push(`model: ${modelName.target}`);

      title = parts.join(", ");
      borderColor = "green";
    }
    const debugFiles = [
      photo?.debug,
      modelType?.debug,
      modelName?.debug,
    ].filter(Boolean);

    return {
      ...item,
      title,
      borderColor,
      debugFiles,
    };
  });
}
