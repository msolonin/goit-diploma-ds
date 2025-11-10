import React, { useState } from "react";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

export default function BoatUploader() {
  const [files, setFiles] = useState([]);
  const [debug, setDebug] = useState(false);
  const [showNextStep, setShowNextStep] = useState(false);

  const [boatType, setBoatType] = useState("motor");
  const [boatName, setBoatName] = useState("");
  const [description, setDescription] = useState("");

  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFiles([...e.target.files]);
  };

  const handleNext = async () => {
    if (files.length === 0) return;

    setLoading(true);

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    formData.append("debug", debug);

    try {
      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      // Real-time simulation: append each file's result
      setResults(data.files || []); // assuming backend returns { files: [{ filename, analysis: {...} }] }
      setShowNextStep(true);
    } catch (error) {
      console.error("Error uploading files:", error);
      toast.error("Failed to analyze files!");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = () => {
    if (results.length === 0) {
      toast.error("No analysis results available!");
      return;
    }

    // Check if at least one winner model exists
    const hasWinner = results.some((fileResult) => fileResult.analysis.winner_model);

    if (hasWinner) {
      toast.success("✅ Model predicted successfully!");
    } else {
      toast.error("❌ No model could be predicted.");
    }

    // Here you can also send boatType, boatName, description, and results to backend
    console.log({ boatType, boatName, description, results });
  };

  return (
    <div style={{ maxWidth: "700px", margin: "20px auto" }}>
      <h2>Boat Uploader</h2>

      <input type="file" multiple onChange={handleFileChange} accept="image/*" />
      <div>
        <label>
          <input type="checkbox" checked={debug} onChange={(e) => setDebug(e.target.checked)} /> Debug
        </label>
      </div>

      {files.length > 0 && !showNextStep && (
        <button onClick={handleNext} style={{ marginTop: "10px" }}>
          {loading ? "Uploading..." : "Next"}
        </button>
      )}

      {results.length > 0 && (
        <div style={{ marginTop: "20px" }}>
          <h3>Analysis Results</h3>
          {results.map((fileResult) => (
            <div key={fileResult.filename} style={{ marginBottom: "15px", padding: "10px", border: "1px solid #ccc" }}>
              <strong>{fileResult.filename}</strong>
              <div>
                <strong>Photo Type Counts:</strong> {JSON.stringify(fileResult.analysis.photo_type_counts)}
              </div>
              <div>
                <strong>Winner Model:</strong>{" "}
                {fileResult.analysis.winner_model
                  ? `${fileResult.analysis.winner_model.model_type} / ${fileResult.analysis.winner_model.model_name} (score: ${fileResult.analysis.winner_model.avg_score}, votes: ${fileResult.analysis.winner_model.votes})`
                  : "None"}
              </div>
            </div>
          ))}
        </div>
      )}

      {showNextStep && (
        <div style={{ marginTop: "20px" }}>
          <div>
            <label>
              Boat Type:
              <select value={boatType} onChange={(e) => setBoatType(e.target.value)}>
                <option value="motor">Motor</option>
                <option value="seal">Seal</option>
              </select>
            </label>
          </div>

          <div style={{ marginTop: "10px" }}>
            <label>
              Boat Name:
              <input type="text" value={boatName} onChange={(e) => setBoatName(e.target.value)} />
            </label>
          </div>

          <div style={{ marginTop: "10px" }}>
            <label>
              Description:
              <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={5} style={{ width: "100%" }} />
            </label>
          </div>

          <button style={{ marginTop: "10px" }} onClick={handleSubmit}>
            Submit
          </button>
        </div>
      )}
      <ToastContainer autoClose={5000} position="top-right" />
    </div>
  );
}
