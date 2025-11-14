import React from "react";
import BoatUploader from "./components/BoatUploader/BoatUploader";

export default function App() {
  return (
    <div>
      <h1 style={{ textAlign: "center", marginTop: "20px" }}>Add new boat</h1>
      <BoatUploader />
    </div>
  );
}
