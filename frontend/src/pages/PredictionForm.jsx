import React, { useState } from "react";
import ReactSelect from "react-select";

import {
  Box,
  FormControl,
  FormLabel,
  Input,
  Button,
  Grid,
  useToast,
} from "@chakra-ui/react";

import data from "../../public/selection_data";

const { categories, jobs, cities } = data;
const categoryOptions = categories.map((category) => ({
  value: category,
  label: category,
}));
const jobOptions = jobs.map((job) => ({ value: job, label: job }));
const cityOptions = cities.map((city) => ({ value: city, label: city }));

function PredictionForm() {
  const [form, setForm] = useState({
    age: "",
    city: "",
    job: "",
    category: "",
    timeOfTransaction: "",
    creditCardNumber: "",
    amount: "",
    merchantLocation: "",
  });

  const toast = useToast();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm({
      ...form,
      [name]: value,
    });
  };

  const handleSelectChange = (selectedOption, fieldName) => {
    setForm({
      ...form,
      [fieldName]: selectedOption.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(form),
      });
      const data = await response.json();
      console.log(data.prediction);

      if (data.prediction && data.prediction === '1') {
        toast({
          title: "Fraud",
          description: "This transaction is fraudulent.",
          status: "error",
          duration: 9000,
          isClosable: true,
          position: "top",
        });
      } else {
        toast({
          title: "Not Fraud",
          description: "This transaction is not fraudulent.",
          status: "success",
          duration: 9000,
          isClosable: true,
          position: "top",
        });
      }
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <Box
      as="form"
      p={200}
      shadow="md"
      borderWidth="1px"
      onSubmit={handleSubmit}
    >
      <Grid templateColumns="repeat(2, 1fr)" gap={6}>
        <FormControl id="age" isRequired>
          <FormLabel>Age</FormLabel>
          <Input name="age" type="number" onChange={handleChange} />
        </FormControl>
        <FormControl id="city" isRequired>
          <FormLabel>City</FormLabel>
          <ReactSelect
            name="city"
            options={cityOptions}
            onChange = {(selectedOption) => handleSelectChange(selectedOption, "city")}
          />
        </FormControl>
        <FormControl id="job" isRequired>
          <FormLabel>Job</FormLabel>
          <ReactSelect
            name="job"
            options={jobOptions}
            onChange = {(selectedOption) => handleSelectChange(selectedOption, "job")}
          />

        </FormControl>
        <FormControl id="category" isRequired>
          <FormLabel>Category</FormLabel>
          <ReactSelect
            name="category"
            options={categoryOptions}
            onChange = {(selectedOption) => handleSelectChange(selectedOption, "category")}
          />
        </FormControl>
        <FormControl id="timeOfTransaction" isRequired>
          <FormLabel>Time of Transaction</FormLabel>
          <Input name="timeOfTransaction" type="time" onChange={handleChange} />
        </FormControl>
        <FormControl id="creditCardNumber" isRequired>
          <FormLabel>Credit Card Number</FormLabel>
          <Input name="creditCardNumber" type="text" onChange={handleChange} />
        </FormControl>
        <FormControl id="amount" isRequired>
          <FormLabel>Amount</FormLabel>
          <Input name="amount" type="decimal" onChange={handleChange} />
        </FormControl>
        <FormControl id="merchantLocation" isRequired>
          <FormLabel>Merchant Location</FormLabel>
          <Input name="merchantLocation" type="text" onChange={handleChange} />
        </FormControl>
      </Grid>
      <Button mt={4} colorScheme="teal" type="submit">
        Submit
      </Button>
    </Box>
  );
}

export default PredictionForm;
