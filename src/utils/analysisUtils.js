export const THRESHOLD = 90;
export const MODEL_THRESHOLD = 65;
export const PHOTO_TYPE_WEIGHTS = { boat: 1.0, out: 0.7, in: 0.3 };
export const LOCAL_STORAGE_KEY = "boatFiles";



export function parseAnalysisData() {
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

export function generateTips() {
  const storedData = JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY) || "[]");

  if (storedData.length === 0) return [];

  const tips = [];
  let hasBoat = false;
  let hasIn = false;

  for (const item of storedData) {
    const filename = item.filename;
    const photo = item.analysis?.photo_type;

    if (photo) {
      const type = photo.target;
      const percent = photo.percent ?? 0;

      if (percent < THRESHOLD) {
        tips.push({
          color: "red",
          text: `Unrecognized photo type: ${filename}`,
        });
      }

      if (type === "boat" && percent >= THRESHOLD) hasBoat = true;
      if (type === "in" && percent >= THRESHOLD) hasIn = true;
    }
  }

  if (!hasBoat) {
    tips.push({
      color: "red",
      text: "Add photo with all boat",
    });
  }

  if (!hasIn) {
    tips.push({
      color: "orange",
      text: "Add interior photos",
    });
  }
  return tips;
}

export function analyzePhotoData(data) {
  const photoTypeCounter = {};
  const modelScores = {};
  const modelVotes = {};
  const modelTypes = {};

  data.forEach((res) => {
    const photo = res.analysis?.photo_type;
    const modelType = res.analysis?.model_type;
    const modelName = res.analysis?.model_name;

    if (!photo) return;
    const photoType = photo.target;
    photoTypeCounter[photoType] = (photoTypeCounter[photoType] || 0) + 1;

    // Only "boat" or "out" contribute to model votes
    if (
      photoType !== "in" &&
      modelName &&
      modelName.percent >= MODEL_THRESHOLD
    ) {
      const mType = modelType ? modelType.target : null;
      const mName = modelName.target;
      const score = modelName.percent * (PHOTO_TYPE_WEIGHTS[photoType] || 1.0);

      const key = `${mType}::${mName}`;
      if (!modelScores[key]) modelScores[key] = [];
      if (!modelVotes[key]) modelVotes[key] = [];
      modelScores[key].push(score);
      modelVotes[key].push(1);
      modelTypes[key] = mType;
    }
  });

  // Aggregate model scores
  const finalSummary = Object.keys(modelScores).map((key) => {
    const [mType, mName] = key.split("::");
    const scores = modelScores[key];
    const avgScore =
      scores.reduce((a, b) => a + b, 0) / (scores.length || 1);
    return {
      model_type: mType,
      model_name: mName,
      avg_score: Number(avgScore.toFixed(2)),
      votes: modelVotes[key].length,
    };
  });

  const winner =
    finalSummary.length > 0
      ? finalSummary.reduce((a, b) => (a.avg_score > b.avg_score ? a : b))
      : null;

  return {
    photo_type_counts: photoTypeCounter,
    winner_model: winner,
  };
}


export function hasUnrecognizedPhotos(data) {
  if (!Array.isArray(data) || data.length === 0) return false;
  return data.some((item) => {
    const photo = item.analysis?.photo_type;
    return !photo || photo.percent < THRESHOLD;
  });
}


export function getSeasonFromDates(fromDate, toDate) {
  if (!fromDate || !toDate) return "mid";
  const fromMonth = new Date(fromDate).getMonth() + 1;
  const toMonth = new Date(toDate).getMonth() + 1;
  const avgMonth = Math.round((fromMonth + toMonth) / 2);
  if ([6, 7, 8].includes(avgMonth)) return "high";
  if ([4, 5, 9, 10].includes(avgMonth)) return "mid";
  return "low";
}

export function calculateRentPrice(basePrice, year, season) {
  if (!basePrice || isNaN(basePrice)) return { dayPrice: 0, weekPrice: 0, monthPrice: 0 };
  let seasonMultiplier = 1;
  switch ((season || "mid").toLowerCase()) {
    case "high":
      seasonMultiplier = 1.3;
      break;
    case "mid":
      seasonMultiplier = 1.1;
      break;
    case "low":
      seasonMultiplier = 0.9;
      break;
    default:
      seasonMultiplier = 1;
  }

  const currentYear = new Date().getFullYear();
  const age = currentYear - (year || currentYear);
  let ageMultiplier = 1;
  if (age <= 2) ageMultiplier = 1.2; // very new
  else if (age <= 5) ageMultiplier = 1.1; // moderately new
  else if (age <= 10) ageMultiplier = 1; // normal
  else ageMultiplier = 0.85; // older


  let dayPrice = basePrice * 0.01 * seasonMultiplier * ageMultiplier; // 1% of boat price per day
  let weekPrice = dayPrice * 7 * 0.9; // 10% discount for week
  let monthPrice = dayPrice * 30 * 0.8; // 20% discount for month
  dayPrice = Math.round(dayPrice);
  weekPrice = Math.round(weekPrice);
  monthPrice = Math.round(monthPrice);
  return { dayPrice, weekPrice, monthPrice };
}