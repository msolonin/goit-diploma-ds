import React, { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import {
  generateTips,
  analyzePhotoData,
  hasUnrecognizedPhotos,
  calculateRentPrice,
  getSeasonFromDates
} from "../../utils/analysisUtils";
import { API_BASE_URL } from "../../config";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import {
  Box,
  Button,
  Switch,
  FormControlLabel,
  TextField,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress,
  Paper,
  Grid,
  IconButton,
  Autocomplete,
  Divider
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import DebugPanel from "./DebugPanel";

/* Debounce helper */
function debounce(fn, delay) {
  let timeout;
  return (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => fn(...args), delay);
  };
}

export default function BoatUploader() {
  const [dropzoneFiles, setDropzoneFiles] = useState([]);
  const [debug, setDebug] = useState(false);
  const [results, setResults] = useState([]);
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [showNextStep, setShowNextStep] = useState(false);

  const [boatType, setBoatType] = useState("motor");
  const [boatName, setBoatName] = useState("");
  const [description, setDescription] = useState("");
  const [year, setYear] = useState("");
  const [price, setPrice] = useState("");
  const [currency, setCurrency] = useState("EUR");

  // Today's date and default To date (+1 month)
  const todayDate = new Date();
  const todayStr = todayDate.toISOString().split("T")[0];
  const defaultToDate = new Date(todayDate);
  defaultToDate.setMonth(defaultToDate.getMonth() + 1);
  const defaultToStr = defaultToDate.toISOString().split("T")[0];

  const [rentFrom, setRentFrom] = useState(todayStr);
  const [rentTo, setRentTo] = useState(defaultToStr);

  const [rentPriceDay, setRentPriceDay] = useState("");
  const [rentPriceWeek, setRentPriceWeek] = useState("");
  const [rentPriceMonth, setRentPriceMonth] = useState("");

  const [previews, setPreviews] = useState([]);
  const [tips, setTips] = useState([]);
  const [nameOptions, setNameOptions] = useState([]);

  const LOCAL_STORAGE_KEY = "boatFiles";
  const WINNER_BOAT = "winnerBoatInfo";

  const maxYear = new Date().getFullYear();
  const minYear = maxYear - 50;

  // Clear previous local storage files
  useEffect(() => {
    localStorage.removeItem(LOCAL_STORAGE_KEY);
  }, []);

  useEffect(() => {
    return () => previews.forEach((p) => URL.revokeObjectURL(p.src));
  }, [previews]);

  const fetchNamesDebounced = useCallback(
    debounce(async (type, chars) => {
      if (!chars || chars.length < 4) {
        setNameOptions([]);
        return;
      }
      try {
        const resp = await fetch(
          `${API_BASE_URL}/get_names?boat_type=${encodeURIComponent(type)}&chars=${encodeURIComponent(chars)}`
        );
        const data = await resp.json();
        setNameOptions(data?.status === "success" ? data.boat_names || [] : []);
      } catch {
        setNameOptions([]);
      }
    }, 500),
    []
  );

  useEffect(() => {
    fetchNamesDebounced(boatType, boatName);
  }, [boatType, boatName, fetchNamesDebounced]);

  const updateLocalStorageAdd = (newFiles) => {
    const stored = JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY) || "[]");
    const merged = [...stored];
    newFiles.forEach((fileObj) => {
      if (!merged.find((f) => f.filename === fileObj.filename)) merged.push(fileObj);
    });
    localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(merged));
  };

  const updateLocalStorageDelete = (filename) => {
    const stored = JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY) || "[]");
    const updated = stored.filter((f) => f.filename !== filename);
    localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(updated));
  };

  const onDrop = useCallback(
    (acceptedFiles) => {
      if (!acceptedFiles || acceptedFiles.length === 0) return;
      setDropzoneFiles((prev) => [...prev, ...acceptedFiles]);
      const newPreviews = acceptedFiles.map((file) => ({ file, src: URL.createObjectURL(file) }));
      setPreviews((prev) => [...prev, ...newPreviews]);
      setShowNextStep(false);
      analyzeFiles(acceptedFiles);
    },
    [debug]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [] }
  });

  const analyzeFiles = async (filesToAnalyze) => {
    setLoadingFiles(true);
    const updatedResults = [...results];

    for (const file of filesToAnalyze) {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("debug", debug);

      try {
        const response = await fetch(`${API_BASE_URL}/analyze_file`, {
          method: "POST",
          body: formData
        });
        const data = await response.json();

        if (data?.status === "success") {
          toast.success(`Analyzed: ${file.name}`);
          const fileResult = { filename: file.name, analysis: data.data };
          updatedResults.push(fileResult);
          updateLocalStorageAdd([fileResult]);
        } else toast.error(`Failed: ${file.name}`);
      } catch {
        toast.error(`Error analyzing ${file.name}`);
      }
    }

    setResults(updatedResults);
    setLoadingFiles(false);
  };

  const handleDeleteFile = (fileToDelete) => {
    const updatedPreviews = previews.filter((p) => p.file !== fileToDelete);
    setPreviews(updatedPreviews);
    setDropzoneFiles((prev) => prev.filter((f) => f !== fileToDelete));
    setResults((prev) => prev.filter((r) => r.filename !== fileToDelete.name));
    updateLocalStorageDelete(fileToDelete.name);

    if (updatedPreviews.length === 0) setShowNextStep(false);
    toast.info(`Removed ${fileToDelete.name}`);
  };

  const handleAddFromLocalStorage = async () => {
    const storedData = JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY) || "[]");
    if (!storedData || storedData.length === 0) {
      toast.error("No analyzed files in local storage!");
      return;
    }

    setShowNextStep(true);
    setTips(generateTips());

    const analysisSummary = analyzePhotoData(storedData);
    const detectedModels = storedData.map((f) => f.analysis?.model_name).filter(Boolean);
    const uniqueModels = [...new Set(detectedModels)];

    if (uniqueModels.length > 1) {
      setTips((prev) => [{ text: "Different types of boat recognized", color: "red" }, ...prev]);
      const detectedTypes = storedData.map((f) => f.analysis?.model_type).filter(Boolean).map(t => t.toLowerCase());
      const uniqueTypes = [...new Set(detectedTypes)];
      if (uniqueTypes.length === 1) setBoatType(uniqueTypes[0].includes("seal") ? "seal" : "motor");
      else setBoatType("");
      setBoatName("");
      setDescription("");
      localStorage.removeItem(WINNER_BOAT);
      return;
    }

    if (analysisSummary?.winner_model) {
      const { model_type, model_name } = analysisSummary.winner_model;

      try {
        const endpoint = model_type.toLowerCase().includes("seal")
          ? `${API_BASE_URL}/get_seal?model_name=${encodeURIComponent(model_name)}`
          : `${API_BASE_URL}/get_motor?model_name=${encodeURIComponent(model_name)}`;

        const response = await fetch(endpoint);
        const data = await response.json();

        if (data?.status === "success" && Array.isArray(data.data) && data.data.length > 0) {
          const info = data.data[0];
          const resolvedBoatType = (info.boat_type || model_type).toLowerCase().includes("sail") ? "seal" : "motor";
          setBoatType(resolvedBoatType);
          setBoatName(info.boat_name || model_name);
          setDescription(info.boat_description || "");
          localStorage.setItem(WINNER_BOAT, JSON.stringify({
            model_type,
            model_name,
            boat_name: info.boat_name,
            description: info.boat_description,
            boat_type: info.boat_type
          }));
          toast.success(`Winner detected: ${info.boat_name}`);
        } else {
          setBoatType(model_type.toLowerCase().includes("seal") ? "seal" : "motor");
          setBoatName("");
          setDescription("");
          localStorage.removeItem(WINNER_BOAT);
        }
      } catch {
        setBoatType("");
        setBoatName("");
        setDescription("");
        localStorage.removeItem(WINNER_BOAT);
      }
    } else {
      setBoatType("");
      setBoatName("");
      setDescription("");
      localStorage.removeItem(WINNER_BOAT);
    }
  };

  /* Auto-calculate rent prices */
  useEffect(() => {
    if (year && price && rentFrom && rentTo) {
      const season = getSeasonFromDates(rentFrom, rentTo);
      const { dayPrice, weekPrice, monthPrice } = calculateRentPrice(
        Number(price),
        Number(year),
        season
      );
      setRentPriceDay(dayPrice);
      setRentPriceWeek(weekPrice);
      setRentPriceMonth(monthPrice);
    }
  }, [year, price, rentFrom, rentTo]);

  const handleSubmit = () => {
    const winnerData = JSON.parse(localStorage.getItem(WINNER_BOAT) || "null");
    const storedData = JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY) || "[]");
    const hasUnrecognized = hasUnrecognizedPhotos(storedData);

    const fromDate = new Date(rentFrom);
    const toDate = new Date(rentTo);
    const diffDays = Math.ceil((toDate - fromDate) / (1000 * 60 * 60 * 24));

    const boatModel = winnerData?.boat_name || boatName || "Unknown";

    if (winnerData && boatName && winnerData.boat_name && boatName.trim().toLowerCase() !== winnerData.boat_name.trim().toLowerCase()) {
      toast.error(`Boat "${boatName}" does not match recognized. Goes on moderation.`);
    } else if (winnerData && !hasUnrecognized) {
      toast.success(`Boat ${boatModel} submitted. Rent period: ${diffDays} day(s)`);
    } else if (winnerData && hasUnrecognized) {
      toast.error(`Boat submitted: ${boatModel}, but unrecognized photos present. Goes on moderation.`);
    } else {
      toast.error("Boat goes on moderation");
    }

    setTimeout(() => {
      localStorage.removeItem(WINNER_BOAT);
      localStorage.removeItem(LOCAL_STORAGE_KEY);
      window.location.reload();
    }, 3000);
  };

  const storedData = JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY) || "[]");

  return (
    <Grid container spacing={3} sx={{ mt: 4 }}>
      <Grid item xs={12} md={debug ? 8 : 12}>
        <Box sx={{ maxWidth: 700, mx: "auto" }}>
          <Typography variant="h4" gutterBottom>Upload your boat photo</Typography>
          <FormControlLabel
            control={<Switch checked={debug} onChange={(e) => setDebug(e.target.checked)} />}
            label="Debug"
          />

          <Paper {...getRootProps()} sx={{ border: "2px dashed #ccc", p: 4, textAlign: "center", mb: 2, backgroundColor: isDragActive ? "#f0f0f0" : "inherit", cursor: "pointer" }}>
            <input {...getInputProps()} />
            <Typography>{isDragActive ? "Drop the files here..." : "Drag & drop files or click to select"}</Typography>

            {previews.length > 0 && (
              <Grid container spacing={1} sx={{ mt: 2 }}>
                {previews.map((p, idx) => (
                  <Grid item key={idx} sx={{ position: "relative" }}>
                    <Box component="img" src={p.src} alt={`preview-${idx}`} sx={{ width: 80, height: 80, objectFit: "cover", borderRadius: 1, border: "1px solid #ccc" }} />
                    <IconButton size="small" sx={{ position: "absolute", top: -10, right: -10, backgroundColor: "white", "&:hover": { backgroundColor: "#f0f0f0" } }} onClick={(e) => { e.stopPropagation(); handleDeleteFile(p.file); }}>
                      <CloseIcon fontSize="small" />
                    </IconButton>
                  </Grid>
                ))}
              </Grid>
            )}
          </Paper>

          {loadingFiles && <CircularProgress />}

          {previews.length > 0 && !showNextStep && (
            <Button variant="contained" color="primary" onClick={handleAddFromLocalStorage} sx={{ mb: 2 }}>Add</Button>
          )}

        {showNextStep && tips.length > 0 && (
          <Box
            sx={{
              position: "relative",
              mb: 2,
              p: 2,
              borderRadius: 1,
              border: "1px solid #e0e0e0",
              backgroundColor: "#fafafa",
              "&:before": {
                content: '"Tips"',
                position: "absolute",
                top: -10,
                left: 12,
                backgroundColor: "#fafafa",
                padding: "0 4px",
                fontSize: 12,
                color: "#555",
                fontWeight: 500,
              },
            }}
          >
            {tips.map((tip, idx) => (
              <Typography
                key={idx}
                sx={{ color: tip.color || "#555", fontSize: 14, lineHeight: 1.4, mb: 0.3 }}
              >
                {tip.text}
              </Typography>
            ))}
          </Box>
        )}



          {showNextStep && storedData.length > 0 && (
            <Box mt={3}>
              {/* Boat Type */}
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Boat Type</InputLabel>
                <Select value={boatType} onChange={(e) => setBoatType(e.target.value)} label="Boat Type">
                  <MenuItem value="">Select type</MenuItem>
                  <MenuItem value="motor">Motor Yacht</MenuItem>
                  <MenuItem value="seal">Sailing Yacht</MenuItem>
                </Select>
              </FormControl>

              {/* Boat Name */}
              <Autocomplete
                freeSolo
                options={nameOptions}
                inputValue={boatName}
                onInputChange={(event, newInputValue) => setBoatName(newInputValue)}
                renderInput={(params) => <TextField {...params} label="Boat Name" fullWidth sx={{ mb: 2 }} />}
              />

              {/* Description */}
              <TextField fullWidth label="Description" multiline rows={5} value={description} onChange={(e) => setDescription(e.target.value)} sx={{ mb: 2 }} />

              {/* Divider and Year/Price */}
              <Divider sx={{ my: 2 }} />
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    label="Year"
                    type="number"
                    fullWidth
                    value={year}
                    onChange={(e) => {
                      let val = Number(e.target.value);
                      if (val > maxYear) val = maxYear;
                      if (val < minYear) val = minYear;
                      setYear(val);
                    }}
                    inputProps={{ min: minYear, max: maxYear }}
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Price"
                    type="number"
                    fullWidth
                    value={price}
                    onChange={(e) => setPrice(Math.max(0, Number(e.target.value)))}
                    InputProps={{
                      endAdornment: (
                        <Select value={currency} onChange={(e) => setCurrency(e.target.value)} sx={{ ml: 1, minWidth: 60 }}>
                          <MenuItem value="EUR">€</MenuItem>
                          <MenuItem value="USD">$</MenuItem>
                          <MenuItem value="UAH">₴</MenuItem>
                        </Select>
                      )
                    }}
                  />
                </Grid>
              </Grid>

              {/* Rent Period */}
              <Divider sx={{ my: 2 }}>Rent Period</Divider>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    label="From"
                    type="date"
                    fullWidth
                    value={rentFrom}
                    onChange={(e) => setRentFrom(e.target.value)}
                    InputLabelProps={{ shrink: true }}
                    inputProps={{ min: todayStr }}
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="To"
                    type="date"
                    fullWidth
                    value={rentTo}
                    onChange={(e) => {
                      if (rentFrom && e.target.value <= rentFrom) {
                        toast.error("To date must be after From date");
                        return;
                      }
                      setRentTo(e.target.value);
                    }}
                    InputLabelProps={{ shrink: true }}
                    inputProps={{ min: rentFrom || todayStr }}
                  />
                </Grid>
              </Grid>

              {/* Rent Price Section */}
              <Divider sx={{ my: 2 }}>Rent price of boat</Divider>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={3}><TextField label="1 Day" type="number" fullWidth value={rentPriceDay} onChange={(e) => setRentPriceDay(e.target.value)} /></Grid>
                <Grid item xs={3}><TextField label="1 Week" type="number" fullWidth value={rentPriceWeek} onChange={(e) => setRentPriceWeek(e.target.value)} /></Grid>
                <Grid item xs={3}><TextField label="1 Month" type="number" fullWidth value={rentPriceMonth} onChange={(e) => setRentPriceMonth(e.target.value)} /></Grid>
                <Grid item xs={3}>
                  <Select value={currency} onChange={(e) => setCurrency(e.target.value)} fullWidth>
                    <MenuItem value="EUR">€</MenuItem>
                    <MenuItem value="USD">$</MenuItem>
                    <MenuItem value="UAH">₴</MenuItem>
                  </Select>
                </Grid>
              </Grid>

              <Button variant="contained" color="primary" onClick={handleSubmit} sx={{ mt: 2 }}>Submit</Button>
            </Box>
          )}
        </Box>
      </Grid>

      {debug && <Grid item xs={12} md={4}><DebugPanel /></Grid>}
      <ToastContainer autoClose={2000} position="top-right" />
    </Grid>
  );
}
