import { useState } from "react";
import { useForm, FormProvider } from "react-hook-form";
import { useNavigate } from "react-router-dom";
import {
  Typography,
  Grid,
  Box,
  Alert,
  Button,
  Stack,
  IconButton
} from "@mui/material";
import { LoadingButton } from "@mui/lab";
import DeleteIcon from "@mui/icons-material/Delete";

import TextField from "src/components/TextField";
import { addYacht, getPresignedUrl, uploadFileToR2 } from "src/services/yachts";
import { ROUTES } from "src/navigation/routes";
import BoatUploader from "../../components/BoatUploader/BoatUploader";

const AddYachtPage = () => {
  const navigate = useNavigate();

  const [isUploading, setIsUploading] = useState(false);
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [uploadError, setUploadError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const methods = useForm({
    defaultValues: {
      name: "Mandarina",
      type: "Motor Yacht",
      guests: 11,
      cabins: 4,
      crew: 7,
      length: 29,
      year: 2020,
      model: "Custom",
      country: "Italy",
      baseMarina: "Amalfi Coast",
      description: "Some Description",
      summerLowSeasonPrice: 11000,
      summerHighSeasonPrice: 11000,
      winterLowSeasonPrice: 11000,
      winterHighSeasonPrice: 11000
    }
  });

  const { handleSubmit, register } = methods;

  const handleFileChange = (e) => {
    if (!e.target.files) return;

    const newFiles = Array.from(e.target.files);
    const newPreviews = newFiles.map((file) => URL.createObjectURL(file));

    setFiles((prev) => [...prev, ...newFiles]);
    setPreviews((prev) => [...prev, ...newPreviews]);

    e.target.value = "";
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    if (!droppedFiles.length) return;

    const droppedPreviews = droppedFiles.map((file) =>
      URL.createObjectURL(file)
    );

    setFiles((prev) => [...prev, ...droppedFiles]);
    setPreviews((prev) => [...prev, ...droppedPreviews]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const removeFile = (index) => {
    URL.revokeObjectURL(previews[index]);
    setFiles((prev) => prev.filter((_, i) => i !== index));
    setPreviews((prev) => prev.filter((_, i) => i !== index));
  };

  const onSubmit = async (data) => {
    setIsUploading(true);
    setUploadError(null);

    const upperName = data.name.trim().toUpperCase();

    try {
      const photoUrls = [];

      for (let i = 0; i < files.length; i++) {
        const file = files[i];

        const { uploadUrl, publicUrl } = await getPresignedUrl(
          upperName,
          i,
          file.type
        );

        await uploadFileToR2(uploadUrl, file);
        photoUrls.push(publicUrl);
      }

      const createdYacht = await addYacht({
        ...data,
        name: upperName,
        photos: photoUrls
      });

      if (createdYacht?.id) {
        navigate(ROUTES.YACHT_DETAILS.replace(":id", createdYacht.id));
      }
    } catch (error) {
      console.error(error);
      setUploadError("Saving error. Check your data and try again.");
    } finally {
      setIsUploading(false);
    }
  };

return (
  <div style={{ display: "flex", justifyContent: "center", height: "100vh" }}>
    <BoatUploader />
  </div>
);


};

export default AddYachtPage;
