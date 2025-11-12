import React, { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { generateTips, analyzePhotoData, hasUnrecognizedPhotos } from "../../utils/analysisUtils";
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
  Autocomplete
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import DebugPanel from "./DebugPanel";

// Utility debounce function
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
  const [previews, setPreviews] = useState([]);
  const [tips, setTips] = useState([]);
  const [nameOptions, setNameOptions] = useState([]);

  const LOCAL_STORAGE_KEY = "boatFiles";
  const WINNER_BOAT = "winnerBoatInfo";



  useEffect(() => {
    localStorage.removeItem(LOCAL_STORAGE_KEY);
  }, []);

  useEffect(() => {
    return () => previews.forEach((p) => URL.revokeObjectURL(p.src));
  }, [previews]);

  // Debounced API call for get_names
  const fetchNamesDebounced = useCallback(
    debounce(async (type, chars) => {
      if (chars.length < 4) return setNameOptions([]);
      try {
        const response = await fetch(
          `${API_BASE_URL}/get_names?boat_type=${encodeURIComponent(type)}&chars=${encodeURIComponent(chars)}`
        );
        const data = await response.json();
        setNameOptions(data.status === "success" ? data.boat_names : []);
      } catch {
        setNameOptions([]);
      }
    }, 500), // 500ms debounce
    []
  );

  useEffect(() => {
    fetchNamesDebounced(boatType, boatName);
  }, [boatType, boatName, fetchNamesDebounced]);

  // --- Dropzone and analysis logic ---
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
      if (acceptedFiles.length === 0) return;
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
    accept: { "image/*": [] },
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
          body: formData,
        });
        const data = await response.json();
        if (data.status === "success") {
          toast.success(`Analyzed: ${file.name}`);
          const fileResult = { filename: file.name, analysis: data.data };
          updatedResults.push(fileResult);
          updateLocalStorageAdd([fileResult]);
        } else {
          toast.error(`Failed: ${file.name}`);
        }
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
    setResults(results.filter((r) => r.filename !== fileToDelete.name));
    updateLocalStorageDelete(fileToDelete.name);
    if (updatedPreviews.length === 0) setShowNextStep(false);
    toast.info(`Removed ${fileToDelete.name}`);
  };

  const handleAddFromLocalStorage = async () => {
    const storedData = JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY) || "[]");
    if (!storedData.length) {
      toast.error("No analyzed files in local storage!");
      return;
    }
    setShowNextStep(true);
    setTips(generateTips());
    const analysisSummary = analyzePhotoData(storedData);

    if (analysisSummary.winner_model) {
      const { model_type, model_name } = analysisSummary.winner_model;
      try {
        const endpoint =
          model_type.toLowerCase() === "seal"
            ? `${API_BASE_URL}/get_seal?model_name=${encodeURIComponent(model_name)}`
            : `${API_BASE_URL}/get_motor?model_name=${encodeURIComponent(model_name)}`;

        const response = await fetch(endpoint);
        const data = await response.json();
        if (data.status === "success" && data.data?.length > 0) {
          const info = data.data[0];
          setBoatType(info.boat_type?.toLowerCase().includes("sail") ? "seal" : "motor");
          setBoatName(info.boat_name || model_name);
          setDescription(info.boat_description || "");
          localStorage.setItem(
            "winnerBoatInfo",
            JSON.stringify({ model_type, model_name, boat_name: info.boat_name, description: info.boat_description, boat_type: info.boat_type })
          );
          toast.success(`Winner detected: ${info.boat_name}`);
        } else toast.error("No detailed data found for winner model!");
      } catch {
        toast.error("Failed to fetch winner model details!");
      }
    } else localStorage.removeItem("winnerBoatInfo");
  };

    const handleSubmit = () => {
      const winnerData = JSON.parse(localStorage.getItem("winnerBoatInfo") || "null");
      const storedData = JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY) || "[]");
      const hasUnrecognized = hasUnrecognizedPhotos(storedData);

      // --- Validation: Boat name mismatch ---
      if (winnerData && boatName.trim() && boatName.trim().toLowerCase() !== winnerData.boat_name.trim().toLowerCase()) {
        toast.error(`Boat "${boatName}" from autocomplete not match with recognized: "${winnerData.boat_name}". Boat goes on moderation.`);
      }
      else if (winnerData && !hasUnrecognized) {
        toast.success(`✅ Boat: "${winnerData.boat_name}" submitted`);
      }
      else if (winnerData && hasUnrecognized) {
        toast.error(`Boat: "${winnerData.boat_name}" submitted, but unrecognized photo(s) present. Boat goes on moderation.`);
      }
      else {
        toast.error("Boat goes on moderation");
      }

  setTimeout(() => {
    localStorage.removeItem("winnerBoatInfo");
    localStorage.removeItem(LOCAL_STORAGE_KEY);
    window.location.reload();
  }, 3000);
};


  const storedData = JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY) || "[]");

  return (
    <Grid container spacing={3} sx={{ mt: 4 }}>
      <Grid item xs={12} md={debug ? 8 : 12}>
        <Box sx={{ maxWidth: 700, mx: "auto" }}>
          <Typography variant="h4" gutterBottom>Boat Uploader</Typography>
          <FormControlLabel control={<Switch checked={debug} onChange={(e) => setDebug(e.target.checked)} />} label="Debug" />

          <Paper {...getRootProps()} sx={{ border: "2px dashed #ccc", p: 4, textAlign: "center", mb: 2, backgroundColor: isDragActive ? "#f0f0f0" : "inherit", cursor: "pointer" }}>
            <input {...getInputProps()} />
            <Typography>{isDragActive ? "Drop the files here..." : "Drag & drop files here, or click to select"}</Typography>

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
          {previews.length > 0 && !showNextStep && <Button variant="contained" color="primary" onClick={handleAddFromLocalStorage} sx={{ mb: 2 }}>Add</Button>}

          {showNextStep && tips.length > 0 && (
            <Box sx={{ mb: 2, p: 1.5, borderRadius: 1, backgroundColor: "#fafafa", border: "1px solid #e0e0e0" }}>
              <Typography variant="caption" sx={{ display: "block", fontSize: 16, fontWeight: 500, color: "#555", mb: 0.5 }}>Tips:</Typography>
              {tips.map((tip, idx) => <Typography key={idx} sx={{ color: tip.color, fontSize: 14, lineHeight: 1.4, mb: 0.3 }}>{tip.text}</Typography>)}
            </Box>
          )}

          {showNextStep && storedData.length > 0 && (
            <Box mt={3}>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Boat Type</InputLabel>
                <Select value={boatType} onChange={(e) => setBoatType(e.target.value)}>
                  <MenuItem value="motor">Motor Yacht</MenuItem>
                  <MenuItem value="seal">Sailing yacht</MenuItem>
                </Select>
              </FormControl>

                <Autocomplete
                  freeSolo
                  options={nameOptions}
                  inputValue={boatName}
                  onInputChange={(event, newInputValue) => setBoatName(newInputValue)}
                  onChange={async (event, selectedValue) => {
                    if (!selectedValue) return;

                    setBoatName(selectedValue);

                    try {
                      const endpoint =
                        boatType === "seal"
                          ? `${API_BASE_URL}/get_seal?model_name=${encodeURIComponent(selectedValue)}`
                          : `${API_BASE_URL}/get_motor?model_name=${encodeURIComponent(selectedValue)}`;

                      const response = await fetch(endpoint);
                      const data = await response.json();

                      if (data.status === "success" && data.data?.length > 0) {
                        const info = data.data[0];
                        setDescription(info.boat_description || "");
                      } else {
                        setDescription("");
                      }
                    } catch (err) {
                      console.error("Failed to fetch model description:", err);
                      setDescription("");
                    }
                  }}
                  renderInput={(params) => <TextField {...params} label="Boat Name" fullWidth sx={{ mb: 2 }} />}
                />


              <TextField fullWidth label="Description" multiline rows={5} value={description} onChange={(e) => setDescription(e.target.value)} sx={{ mb: 2 }} />
              <Button variant="contained" color="primary" onClick={handleSubmit}>Submit</Button>
            </Box>
          )}
        </Box>
      </Grid>

      {debug && <Grid item xs={12} md={4}><DebugPanel /></Grid>}
      <ToastContainer autoClose={2000} position="top-right" />
    </Grid>
  );
}
