import React, { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { generateTips, analyzePhotoData, hasUnrecognizedPhotos } from "../../utils/analysisUtils";
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
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import DebugPanel from "./DebugPanel";

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


  const LOCAL_STORAGE_KEY = "boatFiles";
  const WINNER_BOAT = "winnerBoatInfo";

  useEffect(() => {
    // clear local storage on full page refresh
    localStorage.removeItem(LOCAL_STORAGE_KEY);
  }, []);

  useEffect(() => {
    return () => previews.forEach((p) => URL.revokeObjectURL(p.src));
  }, [previews]);

  const updateLocalStorageAdd = (newFiles) => {
    const stored = JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY) || "[]");
    const merged = [...stored];
    newFiles.forEach((fileObj) => {
      if (!merged.find((f) => f.filename === fileObj.filename)) {
        merged.push(fileObj);
      }
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
      const newPreviews = acceptedFiles.map((file) => ({
        file,
        src: URL.createObjectURL(file),
      }));
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
        const response = await fetch("http://localhost:8000/analyze_file", {
          method: "POST",
          body: formData,
        });
        const data = await response.json();

        if (data.status === "success") {
          toast.success(`Analyzed: ${file.name}`);
          const fileResult = {
            filename: file.name,
            analysis: data.data,
          };
          updatedResults.push(fileResult);
          updateLocalStorageAdd([fileResult]);
        } else {
          toast.error(`Failed: ${file.name}`);
        }
      } catch (error) {
//         console.error("Error uploading file:", error);
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

    const updatedResults = results.filter(
      (r) => r.filename !== fileToDelete.name
    );
    setResults(updatedResults);
    updateLocalStorageDelete(fileToDelete.name);
    if (updatedPreviews.length === 0) setShowNextStep(false);
    toast.info(`🗑️ Removed ${fileToDelete.name}`);
  };

    const handleAddFromLocalStorage = async () => {
      const storedData = JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY) || "[]");
      if (storedData.length === 0) {
        toast.error("No analyzed files in local storage!");
        return;
      }

      setShowNextStep(true);
      const newTips = generateTips();
      setTips(newTips);

      const analysisSummary = analyzePhotoData(storedData);
//       console.log("Analysis summary:", analysisSummary);

      if (analysisSummary.winner_model) {
        const { model_type, model_name } = analysisSummary.winner_model;
        try {
          const endpoint =
            model_type.toLowerCase() === "seal"
              ? `http://localhost:8000/get_seal?model_name=${encodeURIComponent(model_name)}`
              : `http://localhost:8000/get_motor?model_name=${encodeURIComponent(model_name)}`;

          const response = await fetch(endpoint);
          const data = await response.json();

          if (data.status === "success" && data.data?.length > 0) {
            const info = data.data[0];
//             console.log("Winner model details:", info);

            // Fill form fields
            setBoatType(
              info.boat_type?.toLowerCase().includes("sail")
                ? "seal"
                : "motor"
            );
            setBoatName(info.boat_name || model_name);
            setDescription(info.boat_description || "");

            // === Save winner info to localStorage ===
            localStorage.setItem(
              "winnerBoatInfo",
              JSON.stringify({
                model_type,
                model_name,
                boat_name: info.boat_name,
                description: info.boat_description,
                boat_type: info.boat_type,
              })
            );

            toast.success(`Winner detected: ${info.boat_name}`);
          } else {
            toast.error("No detailed data found for winner model!");
          }
        } catch (err) {
//           console.error("Error fetching model details:", err);
          toast.error("Failed to fetch winner model details!");
        }
      } else {
//         console.warn("No winner model found in analysis summary");
        localStorage.removeItem("winnerBoatInfo");
      }
    };


    const handleSubmit = () => {
      const winnerData = JSON.parse(localStorage.getItem("winnerBoatInfo") || "null");
      const storedData = JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY) || "[]");

      const hasUnrecognized = hasUnrecognizedPhotos(storedData);
      if (winnerData && !hasUnrecognized) {
        toast.success(`✅ Boat: "${winnerData.boat_name}" submitted`);
      } else if (winnerData && hasUnrecognized) {
        toast.error(
          `Boat: "${winnerData.boat_name}" submitted, but unrecognized photo(s) present. Boat goes on moderation.`
        );
      } else {
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
      {/* Left: uploader */}
      <Grid item xs={12} md={debug ? 8 : 12}>
        <Box sx={{ maxWidth: 700, mx: "auto" }}>
          <Typography variant="h4" gutterBottom>
            Boat Uploader
          </Typography>

          <FormControlLabel
            control={
              <Switch
                checked={debug}
                onChange={(e) => setDebug(e.target.checked)}
              />
            }
            label="Debug"
          />

          {/* Dropzone */}
          <Paper
            {...getRootProps()}
            sx={{
              border: "2px dashed #ccc",
              p: 4,
              textAlign: "center",
              mb: 2,
              backgroundColor: isDragActive ? "#f0f0f0" : "inherit",
              cursor: "pointer",
            }}
          >
            <input {...getInputProps()} />
            {isDragActive ? (
              <Typography>Drop the files here...</Typography>
            ) : (
              <Typography>Drag & drop files here, or click to select</Typography>
            )}

            {previews.length > 0 && (
              <Grid container spacing={1} sx={{ mt: 2 }}>
                {previews.map((p, idx) => (
                  <Grid item key={idx} sx={{ position: "relative" }}>
                    <Box
                      component="img"
                      src={p.src}
                      alt={`preview-${idx}`}
                      sx={{
                        width: 80,
                        height: 80,
                        objectFit: "cover",
                        borderRadius: 1,
                        border: "1px solid #ccc",
                      }}
                    />
                    <IconButton
                      size="small"
                      sx={{
                        position: "absolute",
                        top: -10,
                        right: -10,
                        backgroundColor: "white",
                        "&:hover": { backgroundColor: "#f0f0f0" },
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteFile(p.file);
                      }}
                    >
                      <CloseIcon fontSize="small" />
                    </IconButton>
                  </Grid>
                ))}
              </Grid>
            )}
          </Paper>

          {loadingFiles && <CircularProgress />}

          {previews.length > 0 && !showNextStep && (
            <Button
              variant="contained"
              color="primary"
              onClick={handleAddFromLocalStorage}
              sx={{ mb: 2 }}
            >
              Add
            </Button>
          )}
        {showNextStep && tips.length > 0 && (
          <Box
            sx={{
              mb: 2,
              p: 1.5,
              borderRadius: 1,
              backgroundColor: "#fafafa",
              border: "1px solid #e0e0e0",
            }}
          >
            <Typography
              variant="caption"
              sx={{
                display: "block",
                fontSize: 16,
                fontWeight: 500,
                color: "#555",
                mb: 0.5,
              }}
            >
              Tips:
            </Typography>
            {tips.map((tip, idx) => (
              <Typography
                key={idx}
                sx={{
                  color: tip.color,
                  fontSize: 14,
                  lineHeight: 1.4,
                  mb: 0.3,
                }}
              >
                {tip.text}
              </Typography>
            ))}
          </Box>
        )}

          {showNextStep && storedData.length > 0 && (
            <Box mt={3}>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Boat Type</InputLabel>
                <Select
                  value={boatType}
                  onChange={(e) => setBoatType(e.target.value)}
                >
                  <MenuItem value="motor">Motor Yacht</MenuItem>
                  <MenuItem value="sail">Sailing yacht</MenuItem>
                </Select>
              </FormControl>

              <TextField
                fullWidth
                label="Boat Name"
                value={boatName}
                onChange={(e) => setBoatName(e.target.value)}
                sx={{ mb: 2 }}
              />

              <TextField
                fullWidth
                label="Description"
                multiline
                rows={5}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                sx={{ mb: 2 }}
              />

              <Button variant="contained" color="primary" onClick={handleSubmit}>
                Submit
              </Button>
            </Box>
          )}
        </Box>
      </Grid>

      {/* Right: Debug Panel */}
      {debug && (
        <Grid item xs={12} md={4}>
          <DebugPanel />
        </Grid>
      )}

      <ToastContainer autoClose={2000} position="top-right" />
    </Grid>
  );
}
