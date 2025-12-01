import React, { useState } from "react";
import { Box, Typography, Paper, Divider, Modal, IconButton, Grid } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import { parseAnalysisData } from "../../utils/analysisUtils";

export default function DebugPanel() {
  const parsedData = parseAnalysisData();
  const [selectedImage, setSelectedImage] = useState(null);

  return (
    <Paper
      sx={{
        minHeight: "10px",      // minimum height
        height: "auto",          // natural flexible height
        maxHeight: "100%",       // prevents uncontrolled growth
        p: 2,
        border: "1px solid #ddd",
        overflowY: "auto",
        backgroundColor: "#fafafa",
        maxWidth: "1200px",
        mx: "auto",
      }}
    >
      <Typography variant="h6" gutterBottom>
        Debug Results
      </Typography>
      <Divider sx={{ mb: 2 }} />

      {parsedData.length === 0 ? (
        <Typography color="text.secondary">No analyzed files yet</Typography>
      ) : (
        parsedData.map((item, idx) => (
          <Box
            key={idx}
            sx={{
              mb: 2,
              p: 1,
              border: `5px solid ${item.borderColor}`,
              borderRadius: 2,
              backgroundColor: "white",
            }}
          >
            {/* Title based on thresholds */}
            <Typography
              variant="subtitle2"
              sx={{
                mb: 1,
                wordBreak: "break-word",
                fontWeight: 600,
              }}
            >
              {item.title}
            </Typography>

            {/* Debug images */}
            {item.debugFiles.length > 0 ? (
              <Grid container spacing={1}>
                {item.debugFiles.map((debugFile, i) => {
                  const imgPath = `/heatmap/${debugFile}`;
                  return (
                    <Grid item key={i}>
                      <Box
                        component="img"
                        src={imgPath}
                        alt={debugFile}
                        onClick={() => setSelectedImage(imgPath)}
                        sx={{
                          width: 100,
                          height: 100,
                          borderRadius: 0.5,
                          border: "1px solid #eee",
                          cursor: "pointer",
                          objectFit: "cover",
                          "&:hover": { opacity: 0.8 },
                        }}
                      />
                    </Grid>
                  );
                })}
              </Grid>
            ) : (
              <Typography
                color="text.secondary"
                variant="body2"
                sx={{ fontSize: "0.8rem" }}
              >
                No debug heatmaps available
              </Typography>
            )}
          </Box>
        ))
      )}

      {/* Full-size modal preview */}
      <Modal
        open={!!selectedImage}
        onClose={() => setSelectedImage(null)}
        sx={{ display: "flex", alignItems: "center", justifyContent: "center" }}
      >
        <Box sx={{ position: "relative", outline: "none" }}>
          <IconButton
            onClick={() => setSelectedImage(null)}
            sx={{
              position: "absolute",
              top: 8,
              right: 8,
              backgroundColor: "rgba(255,255,255,0.8)",
              "&:hover": { backgroundColor: "rgba(255,255,255,1)" },
            }}
          >
            <CloseIcon />
          </IconButton>
          <Box
            component="img"
            src={selectedImage}
            alt="Full View"
            sx={{
              maxWidth: "90vw",
              maxHeight: "90vh",
              borderRadius: 2,
              boxShadow: 4,
            }}
          />
        </Box>
      </Modal>
    </Paper>
  );
}
