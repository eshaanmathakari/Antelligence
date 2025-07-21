const { ethers } = require("hardhat");

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying with:", deployer.address);

  // 1. FoodToken
  const FoodToken = await ethers.getContractFactory("FoodToken");
  const food = await FoodToken.deploy(deployer.address);
  await food.deployed();
  console.log("FoodToken ➜", food.address);

  // 2. ColonyMemory
  const ColonyMemory = await ethers.getContractFactory("ColonyMemory");
  const memory = await ColonyMemory.deploy();
  await memory.deployed();
  console.log("ColonyMemory ➜", memory.address);
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
