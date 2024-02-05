void setNodesAndEdgesFromBlenderFile()
{	
	FILE *inFile;

	float x, y, z;
	int id, idNode1, idNode2;
	char fileName[256];
	char directory[] = "./NodesMuscles/";
	strcpy(fileName, "");
	strcat(fileName, directory);
	strcat(fileName, NodesMusclesFileName);
	strcat(fileName, "/Nodes");
	//printf("\n fileName = %s\n", fileName);

	inFile = fopen(fileName,"rb");
	if(inFile == NULL)
	{
		printf("\n Can't open Nodes file.\n");
		exit(0);
	}
	
	fscanf(inFile, "%d", &NumberOfNodes);
	printf("\n NumberOfNodes = %d", NumberOfNodes);
	fscanf(inFile, "%d", &PulsePointNode);
	printf("\n PulsePointNode = %d", PulsePointNode);
	fscanf(inFile, "%d", &UpNode);
	printf("\n UpNode = %d", UpNode);
	fscanf(inFile, "%d", &FrontNode);
	printf("\n FrontNode = %d", FrontNode);
	
	// Allocating memory for the CPU nodes and CPU connections.
	Node = (nodeAtributesStructure*)malloc(NumberOfNodes*sizeof(nodeAtributesStructure));
	LinksPerNode = 20;
	ConnectingNodes = (int*)malloc(NumberOfNodes*LinksPerNode*sizeof(int));
	
	for(int i = 0; i < NumberOfNodes; i++)
	{
		//fscanf(inFile, "%d", &id);
		//fscanf(inFile, "%f", &x);
		//fscanf(inFile, "%f", &y);
		//fscanf(inFile, "%f", &z);
		fscanf(inFile, "%d %f %f %f", &id, &x, &y, &z);
		
		Node[id].position.x = x;
		Node[id].position.y = y;
		Node[id].position.z = z;
		
		Node[i].velocity.y = 0.0;
		Node[i].velocity.x = 0.0;
		Node[i].velocity.z = 0.0;
		
		Node[i].force.y = 0.0;
		Node[i].force.x = 0.0;
		Node[i].force.z = 0.0;
	}

	fclose(inFile);
    
	// Setting the nodes to -1 so you can tell the nodes that where not used.
	for(int i = 0; i < NumberOfNodes; i++)
	{
		for(int j = 0; j < LinksPerNode; j++)
		{
			ConnectingNodes[i*LinksPerNode + j] = -1;
		}	
	}
	
	strcpy(fileName, "");
	strcat(fileName, directory);
	strcat(fileName, NodesMusclesFileName);
	strcat(fileName, "/Muscles");
	//printf("\n fileName = %s\n",  fileName);
	
	inFile = fopen(fileName,"rb");
	if (inFile == NULL)
	{
		printf("\n Can't open Muscles file.\n");
		exit(0);
	}
	
	int used, linkId;
	fscanf(inFile, "%d", &NumberOfMuscles);
	printf("\n NumberOfMuscles = %d", NumberOfMuscles);
	for(int i = 0; i < NumberOfMuscles; i++)
	{
		fscanf(inFile, "%d", &id);
		fscanf(inFile, "%d", &idNode1);
		fscanf(inFile, "%d", &idNode2);
		//printf("\n%d %d %d\n", id, idNode1, idNode2);
		used = 0;
		linkId = 0;
		while(used != 1 && linkId < LinksPerNode)
		{
			if(ConnectingNodes[idNode1*LinksPerNode + linkId] == -1) 
			{
				ConnectingNodes[idNode1*LinksPerNode + linkId] = idNode2;
				used = 1;
			}
			else
			{
				linkId++;
			}
		}
		
		used = 0;
		linkId = 0;
		while(used != 1 && linkId < LinksPerNode)
		{
			if(ConnectingNodes[idNode2*LinksPerNode + linkId] == -1) 
			{
				ConnectingNodes[idNode2*LinksPerNode + linkId] = idNode1;
				used = 1;
			}
			else
			{
				linkId++;
			}
		}
	}
	
	fclose(inFile);
	
	strcpy(fileName, "");
	strcat(fileName,NodesMusclesFileName);
	strcat(fileName,"/Nodes");
	checkNodes();
	
	printf("\n Blender generated nodes and links have been created.");
}

void checkNodes()
{
	float dx, dy, dz, d;
	float averageMinSeperation, minSeperation;
	int nearestNeighbor;
	int flag =0;
	
	averageMinSeperation = 0;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		minSeperation = 10000000.0;
		for(int j = 0; j < NumberOfNodes; j++)
		{
			if(i != j)
			{
				dx = Node[i].position.x - Node[j].position.x;
				dy = Node[i].position.y - Node[j].position.y;
				dz = Node[i].position.z - Node[j].position.z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < minSeperation) 
				{
					minSeperation = d;
				}
			}
		}
		averageMinSeperation += minSeperation;
	}
	averageMinSeperation = averageMinSeperation/NumberOfNodes;
	
	for(int i = 0; i < NumberOfNodes; i++)
	{
		minSeperation = 10000000.0;
		for(int j = 0; j < NumberOfNodes; j++)
		{
			if(i != j)
			{
				dx = Node[i].position.x - Node[j].position.x;
				dy = Node[i].position.y - Node[j].position.y;
				dz = Node[i].position.z - Node[j].position.z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < minSeperation)
				{
					minSeperation = d;
					nearestNeighbor = j;
				}
			}
		}
		if(minSeperation < averageMinSeperation/100.0)
		{
			printf("\n Nodes %d and %d are too close. Their separation is %f", i, nearestNeighbor, minSeperation);
			flag = 1;
		}
	}
	if(flag ==1)
	{
		printf("\n The average nearest seperation for all the nodes is %f.", averageMinSeperation);
		printf("\n The cutoff seperation was %f.\n\n", averageMinSeperation/10.0);
		exit(0);
	}
	printf("\n Nodes have been checked for minimal separation.");
}

int findNumberOfMuscles()
{
	int count = 0;
	
	for(int i = 0; i < NumberOfNodes; i++)
	{
		for(int j = 0; j < LinksPerNode; j++)
		{
			if(ConnectingNodes[i*LinksPerNode + j] != -1 && ConnectingNodes[i*LinksPerNode + j] > i)
			{
				count++;
			}
		}
	}
	return(count);
}

// This code numbers the muscles and connects each end of a muscle to a node.
void linkMusclesToNodes()
{
	int nodeNumberToLinkTo;
	//Setting the ends of the muscles to nodes
	int index = 0;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		for(int j = 0; j < LinksPerNode; j++)
		{
			if(NumberOfNodes*LinksPerNode <= (i*LinksPerNode + j))
			{
				printf("\n TSU Error: number of ConnectingNodes is out of bounds\n");
				exit(0);
			}
			nodeNumberToLinkTo = ConnectingNodes[i*LinksPerNode + j];
			
			if(nodeNumberToLinkTo != -1)
			{
				if(i < nodeNumberToLinkTo)
				{
					if(NumberOfMuscles <= index)
					{
						printf("\n TSU Error: number of muscles is out of bounds index = %d\n", index);
						exit(0);
					} 
					Muscle[index].nodeA = i;
					Muscle[index].nodeB = nodeNumberToLinkTo;
					index++;
				}
			}
		}
	}
	
	// Uncomment this to check to see if the muscles are created correctly.
	/*
	for(int i = 0; i < NumberOfMuscles; i++)
	{
		printf("\n Muscle[%d].nodeA = %d  Muscle[%d].nodeB = %d", i, Muscle[i].nodeA, i, Muscle[i].nodeB);
	}
	*/
	
	printf("\n Muscles have been linked to Nodes");
}

// This code connects the newly numbered muscles to the nodes. The nodes know they are connected but they don't the number of the muscle.
void linkNodesToMuscles()
{	
	int nodeNumber;
	// Each node will have a list of muscles they are attached to.
	for(int i = 0; i < NumberOfNodes; i++)
	{
		for(int j = 0; j < LinksPerNode; j++)
		{
			if(NumberOfNodes*LinksPerNode <= (i*LinksPerNode + j))
			{
				printf("\n TSU Error: number of ConnectingNodes is out of bounds in function linkNodesToMuscles\n");
				exit(0);
			}
			nodeNumber = ConnectingNodes[i*LinksPerNode + j];
			if(nodeNumber != -1)
			{
				for(int k = 0; k < NumberOfMuscles; k++)
				{
					if((Muscle[k].nodeA == i && Muscle[k].nodeB == nodeNumber) || (Muscle[k].nodeA == nodeNumber && Muscle[k].nodeB == i))
					{
						ConnectingMuscles[i*LinksPerNode + j] = k;
					}
				}
			}
			else
			{
				// If the link is not attached to a muscle set it to -1.
				ConnectingMuscles[i*LinksPerNode + j] = -1;
			}
		}
	}
	
	// Uncomment this to check to see if the nodes are connected to the correct muscles.
	/*
	for(int i = 0; i < NumberOfNodes; i++)
	{
		for(int j = 0; j < LinksPerNode; j++)
		{
			printf("\n Node = %d  link = %d linked Muscle = %d", i, j, ConnectingMuscles[i*LinksPerNode + j]);
		}	
	}
	*/
	
	printf("\n Nodes have been linked to muscles");
}

double getLogNormal()
{
	//time_t t;
	// Seading the random number generater.
	//srand((unsigned) time(&t));
	double temp1, temp2;
	double randomNumber;
	int test;
	
	// Getting two uniform random numbers in [0,1]
	temp1 = ((double) rand() / (RAND_MAX));
	temp2 = ((double) rand() / (RAND_MAX));
	test = 0;
	while(test ==0)
	{
		// Getting ride of the end points so now random number is in (0,1)
		if(temp1 == 0 || temp1 == 1 || temp2 == 0 || temp2 == 1) 
		{
			test = 0;
		}
		else
		{
			// Using Box-Muller to get a standard normal random number.
			randomNumber = cos(2.0*PI*temp2)*sqrt(-2 * log(temp1));
			// Creating a log-normal distrobution from the normal randon number.
			randomNumber = exp(randomNumber);
			test = 1;
		}

	}
	return(randomNumber);	
}

void setMuscleAttributesAndNodeMasses()
{	
	float dx, dy, dz, d;
	float sum, totalLengthOfAllMuscles;
	float totalSurfaceAreaUsed, totalMassUsed;
	int count;
	float averageRadius, areaSum, areaAdjustment;
	int k;
	int muscleNumber;
	time_t t;
	int muscleTest, muscleTryCount, muscleTryCountMax;
	
	muscleTryCountMax = 100;
	
	// Seading the random number generater.
	srand((unsigned) time(&t));
	
	// Mogy need to work on this ??????
	// Getting some method of asigning an area to a node so we can get a force from the pressure.
	// We are letting the area be the circle made from the average radius out from a the node in question.
	// This will leave some area left out so we will perportionatly distribute this extra area out to the nodes as well.
	// If shape is a circle first we divide by the number of divsions to get the surface area of a great circler.
	// Then scale by the ratio of the circle compared to a great circle.
	// Circles seem to handle less pressure with this sceam so we downed the pressure by 2/3
	// On the atria1 shape we took out the area that the superiorVenaCava and inferiorVenaCava cover. 
	totalSurfaceAreaUsed = 4.0*PI*RadiusOfAtria*RadiusOfAtria;
	
	// Now we are finding the average radius from a node out to all it's nieghbor nodes.
	// Then setting its area to a circle of half this radius.
	areaSum = 0.0;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		averageRadius = 0.0;
		count = 0;
		for(int j = 0; j < LinksPerNode; j++)
		{
			if(NumberOfNodes*LinksPerNode <= (i*LinksPerNode + j))
			{
				printf("\n TSU Error: number of ConnectingNodes is out of bounds in function setMuscleAttributesAndNodeMasses\n");
				exit(0);
			}
			muscleNumber = ConnectingMuscles[i*LinksPerNode + j];
			if(muscleNumber != -1)
			{
				// The muscle is connected to two nodes. One to me and one to you. Need to find out who you are and not connect to myself.
				k = Muscle[muscleNumber].nodeA;
				if(k == i) k = Muscle[muscleNumber].nodeB;
				dx = Node[k].position.x - Node[i].position.x;
				dy = Node[k].position.y - Node[i].position.y;
				dz = Node[k].position.z - Node[i].position.z;
				averageRadius += sqrt(dx*dx + dy*dy + dz*dz);
				count++;
			}
		}
		if(count != 0) 
		{
			averageRadius /= count; // Getting the average radius; 
			averageRadius /= 2.0; // taking half that radius; 
			Node[i].area = PI*averageRadius*averageRadius; 
		}
		else
		{
			Node[i].area = 0.0; 
		}
		areaSum += Node[i].area;
	}
	
	areaAdjustment = totalSurfaceAreaUsed - areaSum;
	if(0.0 < areaAdjustment)
	{
		for(int i = 0; i < NumberOfNodes; i++)
		{
			if(areaSum < 0.00001)
			{
				printf("\n TSU Error: areaSum is too small (%f)\n", areaSum);
				exit(0);
			}
			else 
			{
				Node[i].area += areaAdjustment*Node[i].area/areaSum;
			}
		}
	}
	
	// Need to work on this Mogy ?????????
	// Setting the total mass used. If it is a sphere it is just the mass of the atria.
	// If shape is a circle first we divide by the number of divsions to get the mass a great circler.
	// Then scale by the ratio of the circle compared to a great circle.
	totalMassUsed = MassOfAtria; 
	// Taking out the mass of the two vena cava holes. It should be the same ration as the ratio of the surface areas.
	totalMassUsed *= totalSurfaceAreaUsed/(4.0*PI*RadiusOfAtria*RadiusOfAtria);
	
	//Finding the length of each muscle and the total length of all muscles.
	totalLengthOfAllMuscles = 0.0;
	for(int i = 0; i < NumberOfMuscles; i++)
	{	
		dx = Node[Muscle[i].nodeA].position.x - Node[Muscle[i].nodeB].position.x;
		dy = Node[Muscle[i].nodeA].position.y - Node[Muscle[i].nodeB].position.y;
		dz = Node[Muscle[i].nodeA].position.z - Node[Muscle[i].nodeB].position.z;
		d = sqrt(dx*dx + dy*dy + dz*dz);
		Muscle[i].naturalLength = d;
		totalLengthOfAllMuscles += d;
	}
	
	// Setting the mass of all muscles.
	for(int i = 0; i < NumberOfMuscles; i++)
	{	
		Muscle[i].mass = totalMassUsed*(Muscle[i].naturalLength/totalLengthOfAllMuscles);
	}
	
	// Setting muscle timing functions
	for(int i = 0; i < NumberOfMuscles; i++)
	{
		Muscle[i].apNode = -1;
		Muscle[i].onOff = 0;
		Muscle[i].dead = 0;
		Muscle[i].timer = 0.0;
		
		muscleTest = 0;
		muscleTryCount = 0;
		while(muscleTest == 0)
		{
			Muscle[i].conductionVelocity = BaseMuscleConductionVelocity;
			Muscle[i].conductionDuration = Muscle[i].naturalLength/Muscle[i].conductionVelocity;
			
			Muscle[i].contractionDuration = BaseMuscleContractionDuration;
			
			Muscle[i].rechargeDuration = BaseMuscleRechargeDuration;
			
			// If it takes the electrical wave longer to cross the muscle than it does to get ready 
			// to fire a muscle could excite itself.
			if(Muscle[i].conductionDuration < Muscle[i].contractionDuration + Muscle[i].rechargeDuration)
			{
				muscleTest = 1;
			}
			
			muscleTryCount++;
			if(muscleTryCountMax < muscleTryCount)
			{
				printf(" \n You have tried to create muscle %d over %d times. You need to reset your muscle timing settings.\n", i, muscleTryCountMax);
				printf(" Good Bye\n");
				exit(0);
			}
		}
	}
	
	// Setting strength functions.
	for(int i = 0; i < NumberOfMuscles; i++)
	{	
		Muscle[i].contractionStrength = MyocyteForcePerMass*Muscle[i].mass;
		
		Muscle[i].relaxedStrength = BaseMuscleRelaxedStrength*Muscle[i].mass;
		
		// Making sure the muscle will not contract too much or get longer when it is suppose to shrink.
		muscleTest = 0;
		muscleTryCount = 0;
		while(muscleTest == 0)
		{
			Muscle[i].compresionStopFraction = BaseMuscleCompresionStopFraction;
			
			if(0.5 < Muscle[i].compresionStopFraction && Muscle[i].compresionStopFraction < 1.0)
			{
				muscleTest = 1;
			}
			
			muscleTryCount++;
			if(muscleTryCountMax < muscleTryCount)
			{
				printf(" \n You have tried to create muscle %d over %d times. You need to reset your muscle contraction length settings.\n", i, muscleTryCountMax);
				printf(" Good Bye\n");
				exit(0);
			}
		}
	}
	
	// Setting the display color of the muscle.
	for(int i = 0; i < NumberOfMuscles; i++)
	{	
		Muscle[i].color.x = ReadyColor.x;
		Muscle[i].color.y = ReadyColor.y;
		Muscle[i].color.z = ReadyColor.z;
		Muscle[i].color.w = 0.0;
	}
	
	for(int i = 0; i < NumberOfNodes; i++)
	{
		Node[i].ablatedYesNo = 0; // Setting all nodes to not ablated.
		Node[i].drawFlag = 0; // This flag will allow you to draw cettain nodes even when the draw nodes flag is set to off. Set it to off to start with.
		
		// Setting all node colors to not ablated (green)
		Node[i].color.x = 0.0;
		Node[i].color.y = 1.0;
		Node[i].color.z = 0.0;
		Node[i].color.w = 0.0;
	}
	
	// Setting the node masses
	for(int i = 0; i < NumberOfNodes; i++)
	{
		sum = 0.0;
		for(int j = 0; j < LinksPerNode; j++)
		{
			if(ConnectingMuscles[i*LinksPerNode + j] != -1)
			{
				sum += Muscle[ConnectingMuscles[i*LinksPerNode + j]].mass;
			}
		}
		Node[i].mass = sum/2.0;
	}
	printf("\n Muscle Attributes And Node Masses have been set");
}

