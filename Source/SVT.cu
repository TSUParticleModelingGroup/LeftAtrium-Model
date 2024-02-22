
// nvcc SVT.cu -o svt -lglut -lm -lGLU -lGL
//To force kill hit "control c" in the window you launched it from.
// -lGL -lm -lX11 -lXrandr -lXi -lXxf86vm -lpthread -ldl

#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <GL/glut.h>
#include <GL/freeglut.h>
//#include <GLFW/glfw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <signal.h>
#include <unistd.h>
using namespace std;

// defines for terminal stuff.
#define BOLD_ON  "\e[1m"
#define BOLD_OFF   "\e[m"

// normal defines.
#define PI 3.141592654
#define BLOCKNODES 256
#define BLOCKMUSCLES 256

FILE* MovieFile;
int* Buffer;
int MovieOn;

// CUDA Globals
dim3 BlockNodes, GridNodes;
dim3 BlockMuscles, GridMuscles;

// Timing globals
float Dt;
float PrintRate;
int DrawRate;
int RecenterRate;
int Pause;

// This is the node that the beat iminates from.
int PulsePointNode;

// Nodes that orient the simulation. 
// If UpNode is up and FrontNode is in the front you should be in the standard view.
int UpNode;
int FrontNode;

int AblateOnOff;
int EctopicBeatOnOff;
int AdjustMuscleOnOff;
int FindNodeOnOff;
int EctopicSingleOnOff;
int MouseFunctionOnOff;
int ViewFlag; // 0 orthoganal, 1 fulstum
int MovieFlag; // 0 movie off, 1 movie on
float HitMultiplier;
int ScrollSpeedToggle;
float ScrollSpeed;

int NodesMusclesFileOrPreviousRunsFile;
char NodesMusclesFileName[256];
char PreviousRunFileName[256];
char ViewName[256] = "no view set";
float LineWidth;
int DrawNodesFlag;
float NodeRadiusAdjustment;
int DrawFrontHalfFlag;

float Viscosity;
float MyocyteForcePerMass;
float BloodPressure;

float BeatPeriod;

float MassOfAtria;
float RadiusOfAtria;

int NumberOfNodes;
int NumberOfMuscles;
int LinksPerNode;
int MaxNumberOfperiodicEctopicEvents;

float BaseMuscleRelaxedStrength;
float BaseMuscleCompresionStopFraction;
float BaseMuscleConductionVelocity;
float BaseMuscleConductionVelocityAdjustmentMultiplier;
float BaseMuscleContractionDuration;
float BaseMuscleContractionDurationAdjustmentMultiplier;
float BaseMuscleRechargeDuration;
float BaseMuscleRechargeDurationAdjustmentMultiplier;
float BaseMuscleContractionStrength;

float4 ReadyColor;
float4 ContractingColor;
float4 RestingColor;
float4 DeadColor;

float BackGroundRed;
float BackGroundGreen;
float BackGroundBlue;

double MouseX, MouseY, MouseZ;
int MouseWheelPos = 0;

struct muscleAtributesStructure
{
	int nodeA;
	int nodeB;    
	int apNode;
	int onOff;
	int dead;
	float timer;
	float mass;
	float naturalLength;
	float relaxedStrength;
	float compresionStopFraction;
	float conductionVelocity;
	float conductionDuration;
	float contractionDuration;
	float contractionStrength;
	float rechargeDuration;
	float4 color;
};

muscleAtributesStructure *Muscle;
muscleAtributesStructure *MuscleGPU;

struct nodeAtributesStructure
{
	float4 position;
	float4 velocity;
	float4 force;
	float mass;
	float area;
	int ablatedYesNo;
	int drawFlag;
	float4 color;
};

nodeAtributesStructure *Node;
nodeAtributesStructure *NodeGPU;

struct ectopicEventStructure
{
	int node;
	float period;
	float time;
};

ectopicEventStructure *EctopicEvents;
ectopicEventStructure *EctopicEventsGPU;

// This is a list of all the nodes a node is connected to. It is biuld in the initial structure and used to setup the nodes and the muscles
// then it is not used anymore.
int *ConnectingNodes;  

// This is a list of the muscles that each node is connected to.
int *ConnectingMuscles;
int *ConnectingMusclesGPU;

// This will hold the center of mass on the GPU so the center of mass can be adjusted on the GPU. 
// This will keep us from having to copy the nodes down and up to do this on the CPU.
//float4 *CenterOfMassGPU;

float PrintTimer;
int DrawTimer; 
int RecenterCount;
double RunTime;
float4 CenterOfSimulation;
float4 AngleOfSimulation;

// Window globals
static int Window;
//GLFWwindow *Window;
int XWindowSize;
int YWindowSize; 
double Near;
double Far;
double EyeX;
double EyeY;
double EyeZ;
double CenterX;
double CenterY;
double CenterZ;
double UpX;
double UpY;
double UpZ;
	
// Prototyping functions
void allocateMemory(int, int);
int findNumberOfMuscles();
void setNodesAndEdgesLine(float);
void setNodesAndEdgesCircle(float, float); 
void setNodesAndEdgesSphere(int, float);
void setNodesAndEdgesAtria1(int, float, float, float);
void setNodesAndEdgesThickAtria(int, float, float, float);
void linkMusclesToNodes();
void linkNodesToMuscles();
void setMuscleAttributesAndNodeMasses();
void setIndividualMuscleAttributes();
void drawPicture();
void hardCodedAblations();
void hardCodedPeriodicEctopicEvents();
void copyNodesMusclesToGPU();
void copyNodesMusclesFromGPU();
void n_body(float);
void terminalPrint();
void setup();
void orthoganialView();
void fulstrumView();
void KeyPressed(unsigned char, int, int);
//void mymouse(int, int, int, int);
//void Display(void);
//void reshape(int, int);
void readSimulationParameters();
void errorCheck(const char*);
void checkNodes();
void centerObject();
float4 findCenterOfMass();
void setView(int);
void adjustView();
// Cuda prototyping
void __global__ getForces(muscleAtributesStructure*, nodeAtributesStructure*, int*, float, int, int, float4, float, float, float);
void __global__ updateNodes(nodeAtributesStructure*, int, int, ectopicEventStructure*, int, muscleAtributesStructure*, int*, float, float);
void __global__ updateMuscles(muscleAtributesStructure*, nodeAtributesStructure*, int*, ectopicEventStructure*, int, int, int, int, float);
void __global__ recenter(nodeAtributesStructure*, int, float4, float4);

#include "./setNodesAndMuscles.h"
#include "./callBackFunctions.h"
#include "./drawFunction.h"
#include "./CUDAFunctions.h"

void readSimulationParameters()
{
	ifstream data;
	string name;
	
	data.open("./simulationSetup");
	
	if(data.is_open() == 1)
	{
		getline(data,name,'=');
		data >> NodesMusclesFileOrPreviousRunsFile;
		
		getline(data,name,'=');
		data >> NodesMusclesFileName;
		
		getline(data,name,'=');
		data >> PreviousRunFileName;
		
		getline(data,name,'=');
		data >> LineWidth;
		
		getline(data,name,'=');
		data >> NodeRadiusAdjustment;
		
		getline(data,name,'=');
		data >> MyocyteForcePerMass;
		
		getline(data,name,'=');
		data >> BloodPressure;
		
		getline(data,name,'=');
		data >> MassOfAtria;
		
		getline(data,name,'=');
		data >> RadiusOfAtria;
		
		getline(data,name,'=');
		data >> BaseMuscleRelaxedStrength;
		
		getline(data,name,'=');
		data >> BaseMuscleCompresionStopFraction;
		
		getline(data,name,'=');
		data >> BaseMuscleContractionDuration;
		
		getline(data,name,'=');
		data >> BaseMuscleRechargeDuration;
		
		getline(data,name,'=');
		data >> BaseMuscleConductionVelocity;
		
		getline(data,name,'=');
		data >> MaxNumberOfperiodicEctopicEvents;
		
		getline(data,name,'=');
		data >> BeatPeriod;
		
		getline(data,name,'=');
		data >> PrintRate;
		
		getline(data,name,'=');
		data >> DrawRate;
		
		getline(data,name,'=');
		data >> Dt;
		
		getline(data,name,'=');
		data >> ReadyColor.x;
		
		getline(data,name,'=');
		data >> ReadyColor.y;
		
		getline(data,name,'=');
		data >> ReadyColor.z;
		
		getline(data,name,'=');
		data >> ContractingColor.x;
		
		getline(data,name,'=');
		data >> ContractingColor.y;
		
		getline(data,name,'=');
		data >> ContractingColor.z;
		
		getline(data,name,'=');
		data >> RestingColor.x;
		
		getline(data,name,'=');
		data >> RestingColor.y;
		
		getline(data,name,'=');
		data >> RestingColor.z;
		
		getline(data,name,'=');
		data >> DeadColor.x;
		
		getline(data,name,'=');
		data >> DeadColor.y;
		
		getline(data,name,'=');
		data >> DeadColor.z;
		
		getline(data,name,'=');
		data >> BackGroundRed;
		
		getline(data,name,'=');
		data >> BackGroundGreen;
		
		getline(data,name,'=');
		data >> BackGroundBlue;
	}
	else
	{
		printf("\nTSU Error could not open simulationSetup file\n");
		exit(0);
	}
	
	/*
	if(NodesMusclesFileOrPreviousRunsFile == 0)
	{
		printf("\n Object Name = %s", NodesMusclesFileName);
	}
	else if(NodesMusclesFileOrPreviousRunsFile == 1)
	{
		printf("\n Object Name = %s", PreviousRunFileName);
	}
	*/
	
	RecenterRate = 10; //Mogy
	
	data.close();
	// Adjusting blood presure from millimeters of Mercury to our units.
	BloodPressure *= 0.000133322387415;
	
	printf("\n Simulation Parameters have been read in.");
}

void allocateMemory()
{
	int numberOfMusclesTest;
	setNodesAndEdgesFromBlenderFile();
	
	numberOfMusclesTest = findNumberOfMuscles();
	if(numberOfMusclesTest != NumberOfMuscles)
	{
		printf("\n\nNumber of muscles do not matchup. Something is wrong!\n\n");
		//printf("%d != %d\n\n", numberOfMusclesTest, NumberOfMuscles);
		exit(0);
	}
	Muscle = (muscleAtributesStructure*)malloc(NumberOfMuscles*sizeof(muscleAtributesStructure));
	ConnectingMuscles = (int*)malloc(NumberOfNodes*LinksPerNode*sizeof(int));
	linkMusclesToNodes();
	linkNodesToMuscles();
	
	printf("\n number of nodes = %d", NumberOfNodes);
	printf("\n number of muscles = %d", NumberOfMuscles);
	printf("\n number of links per node = %d", LinksPerNode);
	
	BlockNodes.x = BLOCKNODES;
	BlockNodes.y = 1;
	BlockNodes.z = 1;
	
	GridNodes.x = (NumberOfNodes - 1)/BlockNodes.x + 1;
	GridNodes.y = 1;
	GridNodes.z = 1;
	
	BlockMuscles.x = BLOCKMUSCLES;
	BlockMuscles.y = 1;
	BlockMuscles.z = 1;
	
	GridMuscles.x = (NumberOfMuscles - 1)/BlockMuscles.x + 1;
	GridMuscles.y = 1;
	GridMuscles.z = 1;
	
	//CPU memory is allocated in setNodesAndMuscles.h
	cudaMalloc((void**)&MuscleGPU, NumberOfMuscles*sizeof(muscleAtributesStructure));
	errorCheck("cudaMalloc MuscleGPU");
	//CPU memory is allocated setNodesAndMuscles.h
	cudaMalloc((void**)&NodeGPU, NumberOfNodes*sizeof(nodeAtributesStructure));
	errorCheck("cudaMalloc NodeGPU");
	//CPU memory is allocated setNodesAndMuscles.h
	cudaMalloc((void**)&ConnectingMusclesGPU, NumberOfNodes*LinksPerNode*sizeof(int));
	errorCheck("cudaMalloc ConnectingMusclesGPU");
	// Allocating memory for the ectopic events then setting everything to -1 so we can see that they have not been turned on.
	EctopicEvents = (ectopicEventStructure*)malloc(MaxNumberOfperiodicEctopicEvents*sizeof(ectopicEventStructure));
	cudaMalloc((void**)&EctopicEventsGPU, MaxNumberOfperiodicEctopicEvents*sizeof(ectopicEventStructure));
	errorCheck("cudaMalloc EctopicEventsGPU");
	
	for(int i = 0; i < MaxNumberOfperiodicEctopicEvents; i++)
	{
		EctopicEvents[i].node = -1;
		EctopicEvents[i].period = -1.0;
		EctopicEvents[i].time = -1.0;
	}
	printf("\n Memory has been allocated");
}

void setIndividualMuscleAttributes()
{
	int start = 0;
	int stop = 0;
	int index = 0;
	
	if(NumberOfMuscles < start || NumberOfMuscles < stop || NumberOfMuscles < index)
	{
		printf("\n Stop or index is out of range.");
		printf("\n Good Bye \n");
		exit(0);
	}
	// To set individual muscles atribures follow the giuld below. This works on muscle index. Change the 1.0s to what ever you want.
	/*
	index = 1;
	Muscle[index].conductionVelocity = BaseMuscleConductionVelocity*(1.0);
	Muscle[index].conductionDuration = Muscle[index].naturalLength/Muscle[index].conductionVelocity;
	Muscle[index].contractionDuration = BaseMuscleContractionDuration*(1.0);
	Muscle[index].rechargeDuration = BaseMuscleRechargeDuration*(1.0);
	Muscle[index].contractionStrength = MyocyteForcePerMass*Muscle[index].mass*(1.0);
	*/
	// To change a sequential of muscles follow the guide below.
	/*
	for(int i = start; i < stop; i++)
	{
		Muscle[i].conductionVelocity = BaseMuscleConductionVelocity*(0.03);
		Muscle[i].conductionDuration = Muscle[i].naturalLength/Muscle[i].conductionVelocity;
		Muscle[i].contractionDuration = BaseMuscleContractionDuration*(1.0);
		Muscle[i].rechargeDuration = BaseMuscleRechargeDuration*(1.0);
		Muscle[i].contractionStrength = MyocyteForcePerMass*Muscle[i].mass*(1.0);
	}
	*/
	
	// Checking to see if the conduction wave leaves the muscle before it can reset.
	// If not a muscle could reset itself.
	for(int i = 0; i < NumberOfMuscles; i++)
	{	
		if((Muscle[i].contractionDuration + Muscle[i].rechargeDuration) < Muscle[i].contractionDuration)
		{
		 	printf("\n Conduction duration is shorter than the (contraction plus recharge) duration in muscle number %d", i);
		 	printf("\nThis muscle will be killed. \n");
		 	Muscle[i].dead = 1;
		 	Muscle[i].color.x = DeadColor.x;
			Muscle[i].color.y = DeadColor.y;
			Muscle[i].color.z = DeadColor.z;
			Muscle[i].color.w = 1.0;
		} 
		
		if(Muscle[i].contractionStrength < Muscle[i].relaxedStrength)
		{
		 	printf("\n The relaxed repultion strenrth of muscle %d is greater than its contraction strength. Rethink your parameters", i);
		 	printf("\n Good Bye \n");
		 	exit(0);
		} 
	}
}

void hardCodedAblations()
{	
	// Note start and index must be lass than NumberOfNodes and stop most be less than or equal to NumberOfNodes.
	
	// To ablate a slected string of nodes set start and stop and uncomment this for loop.
	/*
	int start = ??;
	int stop = ??;
	for(int i = start; i < stop; i++)
	{	
		Node[i].ablatedYesNo = 1;
		Node[i].drawFlag = 1;
		Node[i].color.x = 1.0;
		Node[i].color.y = 1.0;
		Node[i].color.z = 1.0;
	}
	*/
	
	// To ablate a slected node set your index and uncomment this line.
	/*
	int index = ??;
	Node[index].ablatedYesNo = 1;
	Node[index].drawFlag = 1;
	Node[index].color.x = 1.0;
	Node[index].color.y = 1.0;
	Node[index].color.z = 1.0;
	*/
}

void hardCodedPeriodicEctopicEvents()
{	
	// This is the sinus beat.
	EctopicEvents[0].node = PulsePointNode;
	EctopicEvents[0].period = BeatPeriod;
	EctopicEvents[0].time = BeatPeriod;
	
	// To set a recurrent ectopic event set your index(node) and event(which ectopic event) and uncomment the following lines.
	// Note event must be less than the MaxNumberOfperiodicEctopicEvents and don't use event = 0, that is reserved for the sinus beat.
	/*
	event = ???; // Don't use 0 that is reserved for the sinus beat.
	int index = ???; 
	if(Node[index].ablatedYesNo != 1)
	{
		EctopicEvents[event].node = index;
		EctopicEvents[event].period = 10.0;
		EctopicEvents[event].time = EctopicEvents[event].period; // This will make it start right now.
		Node[index].drawFlag = 1;
		Node[index].color.x = 1.0;
		Node[index].color.y = 0.0;
		Node[index].color.z = 1.0;
	}
	*/
	
	// If you want to setup a random set of ectopic beats remove the ???s and uncomment these lines.
	
	int id;
	int numberOfRandomEctopicBeats = 0; // Must be less than the MaxNumberOfperiodicEctopicEvents.
	float beatPeriodUpperBound = 1000;
	time_t t;
	srand((unsigned) time(&t));
	
	if(MaxNumberOfperiodicEctopicEvents < numberOfRandomEctopicBeats)
	{
		printf("\n Your number of random beats is large than the total number of ectopic beats chosen in the setup file.");
		printf("\n Your number of random beats will be set to the max number of actopic beats");
		numberOfRandomEctopicBeats = MaxNumberOfperiodicEctopicEvents;
	}
	for(int i = 1; i < numberOfRandomEctopicBeats + 1; i++) // Must start at 1 because 0 is the sinus beat.
	{
		id = ((float)rand()/(float)RAND_MAX)*NumberOfNodes;
		EctopicEvents[i].node = id;
		Node[id].drawFlag = 1;
		if(Node[id].ablatedYesNo != 1)
		{
			Node[id].color.x = 0.69;
			Node[id].color.y = 0.15;
			Node[id].color.z = 1.0;
		}
		
		EctopicEvents[i].period = ((float)rand()/(float)RAND_MAX)*beatPeriodUpperBound;
		EctopicEvents[i].time = ((float)rand()/(float)RAND_MAX)*EctopicEvents[i].period;
		printf("\nectopic event %d node = %d period = %f, time = %f\n", i, EctopicEvents[i].node, EctopicEvents[i].period, EctopicEvents[i].time);
	}

}

float4 findCenterOfMass()
{
	float4 centerOfMass;
	
	centerOfMass.x = 0.0;
	centerOfMass.y = 0.0;
	centerOfMass.z = 0.0;
	centerOfMass.w = 0.0;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		 centerOfMass.x += Node[i].position.x*Node[i].mass;
		 centerOfMass.y += Node[i].position.y*Node[i].mass;
		 centerOfMass.z += Node[i].position.z*Node[i].mass;
		 centerOfMass.w += Node[i].mass;
	}
	if(centerOfMass.w < 0.00001)
	{
		printf("\n Mass is too small\n");
		printf("\nw Good Bye\n");
		exit(0);
	}
	else
	{
		centerOfMass.x /= centerOfMass.w;
		centerOfMass.y /= centerOfMass.w;
		centerOfMass.z /= centerOfMass.w;
	}
	return(centerOfMass);
}

void centerObject()
{
	float4 centerOfMass = findCenterOfMass();
	for(int i = 0; i < NumberOfNodes; i++)
	{
		Node[i].position.x -= centerOfMass.x;
		Node[i].position.y -= centerOfMass.y;
		Node[i].position.z -= centerOfMass.z;
	}
	CenterOfSimulation.x = 0.0;
	CenterOfSimulation.y = 0.0;
	CenterOfSimulation.z = 0.0;
}

void copyNodesMusclesToGPU()
{
	cudaMemcpy( MuscleGPU, Muscle, NumberOfMuscles*sizeof(muscleAtributesStructure), cudaMemcpyHostToDevice );
	errorCheck("cudaMemcpy Muscle up");
	cudaMemcpy( NodeGPU, Node, NumberOfNodes*sizeof(nodeAtributesStructure), cudaMemcpyHostToDevice );
	errorCheck("cudaMemcpy Node up");
}

void copyNodesMusclesFromGPU()
{
	cudaMemcpy( Muscle, MuscleGPU, NumberOfMuscles*sizeof(muscleAtributesStructure), cudaMemcpyDeviceToHost);
	errorCheck("cudaMemcpy Muscle down");
	cudaMemcpy( Node, NodeGPU, NumberOfNodes*sizeof(nodeAtributesStructure), cudaMemcpyDeviceToHost);
	errorCheck("cudaMemcpy Node down");
}

void n_body(float dt)
{	
	if(Pause != 1)
	{	
		getForces<<<GridNodes, BlockNodes>>>(MuscleGPU, NodeGPU, ConnectingMusclesGPU, dt, NumberOfNodes, LinksPerNode, CenterOfSimulation, BaseMuscleCompresionStopFraction, RadiusOfAtria, BloodPressure);
		errorCheck("getForces");
		cudaDeviceSynchronize();
		
		updateNodes<<<GridNodes, BlockNodes>>>(NodeGPU, NumberOfNodes, LinksPerNode, EctopicEventsGPU, MaxNumberOfperiodicEctopicEvents, MuscleGPU, ConnectingMusclesGPU, dt, RunTime);
		errorCheck("updateNodes");
		cudaDeviceSynchronize();
		
		updateMuscles<<<GridMuscles, BlockMuscles>>>(MuscleGPU, NodeGPU, ConnectingMusclesGPU, EctopicEventsGPU, NumberOfMuscles, NumberOfNodes, LinksPerNode, MaxNumberOfperiodicEctopicEvents, dt, ReadyColor, ContractingColor, RestingColor);
		errorCheck("updateMuscles");
		cudaDeviceSynchronize();
		
		RecenterCount++;
		if(RecenterCount == RecenterRate) 
		{
			float4 centerOfMass;
			centerOfMass.x = 0.0;
			centerOfMass.y = 0.0;
			centerOfMass.z = 0.0;
			centerOfMass.w = 0.0;
			recenter<<<1, BlockNodes.x>>>(NodeGPU, NumberOfNodes, centerOfMass, CenterOfSimulation);
			errorCheck("recenterGPU");
			RecenterCount = 0;
		}
		
		DrawTimer++;
		if(DrawTimer == DrawRate) 
		{
			copyNodesMusclesFromGPU();
			drawPicture();
			DrawTimer = 0;
		}
		
		PrintTimer += dt;
		if(PrintRate <= PrintTimer) 
		{
			terminalPrint();
			PrintTimer = 0.0;
		}
		
		RunTime += dt; 
	}
	else
	{
		drawPicture();
	}
}

void terminalPrint()
{
	system("clear");
	//printf("\033[0;34m"); // blue.
	//printf("\033[0;36m"); // cyan
	//printf("\033[0;33m"); // yellow
	//printf("\033[0;31m"); // red
	//printf("\033[0;32m"); // green
	printf("\033[0m"); // back to white.
	
	printf("\n");
	printf("\n **************************** Simulation Stats ****************************");
	if(Pause == 1) 
	{
		printf("\n The simulation is");
		printf("\033[0;31m");
		printf(BOLD_ON " paused." BOLD_OFF);
	}
	else 
	{
		printf("\n The simulation is");
		printf("\033[0;32m");
		printf(BOLD_ON " running." BOLD_OFF);
	}
	printf("\n Time = %7.2f milliseconds", RunTime);
	printf("\n You are in");
	printf("\033[0;36m");
	printf(BOLD_ON " %s", ViewName);
	printf("\033[0m" BOLD_OFF);
	printf(" view");
	
	if(AblateOnOff == 1) printf("\n You are in ablation mode.");
	else if(EctopicBeatOnOff == 1) printf("\n You are in create ectopic beat mode.");
	else if(AdjustMuscleOnOff == 1) 
	{
		printf("\n You are in adjust muscle mode");
		printf("\n Base muscle contraction multiplier is = %f", BaseMuscleContractionDurationAdjustmentMultiplier);
		printf("\n Base muscle recharge multiplier is = %f", BaseMuscleRechargeDurationAdjustmentMultiplier);
		printf("\n Base muscle electrical conduction speed multiplier is = %f", BaseMuscleConductionVelocityAdjustmentMultiplier);
	}
	else if(FindNodeOnOff == 1) printf("\n You are in find node number mode.");
	else if(EctopicSingleOnOff == 1) printf("\n You are in one off ectopic event mode.");
	else printf("\n There are no mouse functions currently set.");
	
	printf("\n Driving beat node is %d \n The beat rate %f milliseconds.", EctopicEvents[0].node, EctopicEvents[0].period);
	for(int i = 1; i < MaxNumberOfperiodicEctopicEvents; i++)
	{
		if(EctopicEvents[i].node != -1)
		{
			printf("\n Ectopic beat: Node %d Rate of %f milliseconds.", EctopicEvents[i].node, EctopicEvents[i].period);
		}
	}
	
	printf("\n");
	printf("\n **************************** Terminal Comands ****************************");
	printf("\n h: Help");
	printf("\n c: Recenter View");
	printf("\n S: Screenshot");
	printf("\n k: Save Current Run");
	printf("\n B: Lengthen Beat");
	printf("\n b: Shorten Beat");
	printf("\n ?: Find Front and Top Nodes");
	printf("\n");
	
	printf("\n Toggles");
	printf("\n r: Run/Pause            - ");
	if (Pause == 0) printf(BOLD_ON "Running" BOLD_OFF); else printf(BOLD_ON "Paused" BOLD_OFF);
	printf("\n g: Front/Full           - ");
	if (DrawFrontHalfFlag == 0) printf(BOLD_ON "Full" BOLD_OFF); else printf(BOLD_ON "Front" BOLD_OFF);
	printf("\n n: Nodes Off/Half/Full  - ");
	if (DrawNodesFlag == 0) printf(BOLD_ON "Off" BOLD_OFF); else if (DrawNodesFlag == 1) printf(BOLD_ON "Half" BOLD_OFF); else printf(BOLD_ON "Full" BOLD_OFF);
	printf("\n v: Orthogonal/Frustum   - ");
	if (ViewFlag == 0) printf(BOLD_ON "Orthogonal" BOLD_OFF); else printf(BOLD_ON "Frustrum" BOLD_OFF);
	printf("\n m: Movie On/Off         - ");
	if (MovieFlag == 0) printf(BOLD_ON "Movie not recording" BOLD_OFF); else printf(BOLD_ON "Movie recording in progress" BOLD_OFF);
	printf("\n");
	printf("\n Views" );
	printf("\n 7 8 9 | LL  SUP RL" );
	printf("\n 4 5 6 | PA  INF Ref" );
	printf("\n 1 2 3 | LOA AP  ROA" );
	printf("\n");
	printf("\n Adjust views");
	printf("\n w/s: CCW/CW x-axis");
	printf("\n d/a: CCW/CW y-axis");
	printf("\n z/Z: CCW/CW z-axis");
	printf("\n e/E: In/Out Zoom");
	printf("\n");
	printf("\n Set Mouse actions");
	
	printf("\n !: Ablate            - ");
	if (AblateOnOff == 1) 
	{
		printf("\033[0;36m");
		printf(BOLD_ON "On" BOLD_OFF); 
	}
	else printf(BOLD_ON "Off" BOLD_OFF);
	
	printf("\n @: Ectoic Beat       - ");
	if (EctopicBeatOnOff == 1) 
	{
		printf("\033[0;36m");
		printf(BOLD_ON "On" BOLD_OFF); 
	}
	else printf(BOLD_ON "Off" BOLD_OFF);
	
	printf("\n #: Ectopic Trigger   - ");
	if (EctopicSingleOnOff == 1) 
	{
		printf("\033[0;36m");
		printf(BOLD_ON "On" BOLD_OFF); 
	}
	else printf(BOLD_ON "Off" BOLD_OFF);
	
	printf("\n $: Muscle Adjustment - ");
	if (AdjustMuscleOnOff == 1) 
	{
		printf("\033[0;36m");
		printf(BOLD_ON "On" BOLD_OFF); 
	}
	else printf(BOLD_ON "Off" BOLD_OFF);
	
	printf("\n ^: Identify Node     - ");
	if (FindNodeOnOff == 1) 
	{
		printf("\033[0;36m");
		printf(BOLD_ON "On" BOLD_OFF); 
	}
	else printf(BOLD_ON "Off" BOLD_OFF);
	
	printf("\n ): Turns all Mouse functions off");
	printf("\n");
	printf("\n [/]: (left/right bracket) Increase/Decrease mouse selection area");
	printf("\n Selection area = %f times the radius of atrium. \n", HitMultiplier);
	printf("\n ********************************************************************");
	printf("\n");
}

void setup()
{	
	readSimulationParameters();
	if(NodesMusclesFileOrPreviousRunsFile == 0)
	{
		allocateMemory();
		setMuscleAttributesAndNodeMasses();
		setIndividualMuscleAttributes();
		hardCodedAblations();
		hardCodedPeriodicEctopicEvents();
	}
	else if(NodesMusclesFileOrPreviousRunsFile == 1)
	{
		FILE *inFile;
		char fileName[256];
		
		strcpy(fileName, "");
		strcat(fileName,"./PreviousRunsFile/");
		strcat(fileName,PreviousRunFileName);
		strcat(fileName,"/run");
		//printf("\n fileName = %s\n", fileName);

		inFile = fopen(fileName,"rb");
		if(inFile == NULL)
		{
			printf(" Can't open %s file.\n", fileName);
			exit(0);
		}
		
		fread(&PulsePointNode, sizeof(int), 1, inFile);
		printf("\n PulsePointNode = %d", PulsePointNode);
		fread(&UpNode, sizeof(int), 1, inFile);
		printf("\n UpNode = %d", UpNode);
		fread(&FrontNode, sizeof(int), 1, inFile);
		printf("\n FrontNode = %d", FrontNode);
		
		fread(&NumberOfNodes, sizeof(int), 1, inFile);
		printf("\n NumberOfNodes = %d", NumberOfNodes);
		fread(&NumberOfMuscles, sizeof(int), 1, inFile);
		printf("\n NumberOfMuscles = %d", NumberOfMuscles);
		fread(&LinksPerNode, sizeof(int), 1, inFile);
		printf("\n LinksPerNode = %d", LinksPerNode);
		fread(&MaxNumberOfperiodicEctopicEvents, sizeof(int), 1, inFile);
		printf("\n MaxNumberOfperiodicEctopicEvents = %d", MaxNumberOfperiodicEctopicEvents);
		
		Node = (nodeAtributesStructure*)malloc(NumberOfNodes*sizeof(nodeAtributesStructure));
		Muscle = (muscleAtributesStructure*)malloc(NumberOfMuscles*sizeof(muscleAtributesStructure));
		ConnectingMuscles = (int*)malloc(NumberOfNodes*LinksPerNode*sizeof(int));
		EctopicEvents = (ectopicEventStructure*)malloc(MaxNumberOfperiodicEctopicEvents*sizeof(ectopicEventStructure));
		
		cudaMalloc((void**)&MuscleGPU, NumberOfMuscles*sizeof(muscleAtributesStructure));
		errorCheck("cudaMalloc MuscleGPU");
		cudaMalloc((void**)&NodeGPU, NumberOfNodes*sizeof(nodeAtributesStructure));
		errorCheck("cudaMalloc NodeGPU");
		cudaMalloc((void**)&ConnectingMusclesGPU, NumberOfNodes*LinksPerNode*sizeof(int));
		errorCheck("cudaMalloc ConnectingMusclesGPU");
		cudaMalloc((void**)&EctopicEventsGPU, MaxNumberOfperiodicEctopicEvents*sizeof(ectopicEventStructure));
		errorCheck("cudaMalloc EctopicEventsGPU");
	
		for(int i = 0; i < MaxNumberOfperiodicEctopicEvents; i++)
		{
			EctopicEvents[i].node = -1;
			EctopicEvents[i].period = -1.0;
			EctopicEvents[i].time = -1.0;
		}
		printf("\n Memory has been allocated");
	
		
		fread(Node, sizeof(nodeAtributesStructure), NumberOfNodes, inFile);
	  	fread(Muscle, sizeof(muscleAtributesStructure), NumberOfMuscles, inFile);
	  	fread(ConnectingMuscles, sizeof(int), NumberOfNodes*LinksPerNode, inFile);
	  	fread(EctopicEvents, sizeof(ectopicEventStructure), MaxNumberOfperiodicEctopicEvents, inFile);
		fclose(inFile);
		printf("\n Nodes and Muscles have been read in.");
		
		BlockNodes.x = BLOCKNODES;
		BlockNodes.y = 1;
		BlockNodes.z = 1;
		
		GridNodes.x = (NumberOfNodes - 1)/BlockNodes.x + 1;
		GridNodes.y = 1;
		GridNodes.z = 1;
		
		BlockMuscles.x = BLOCKMUSCLES;
		BlockMuscles.y = 1;
		BlockMuscles.z = 1;
		
		GridMuscles.x = (NumberOfMuscles - 1)/BlockMuscles.x + 1;
		GridMuscles.y = 1;
		GridMuscles.z = 1;
	}
	else
	{
		printf("\n Bad NodesMusclesFileOrPreviousRunsFile type.");
		printf("\n Good Bye.");
		exit(0);
	}
	
	AngleOfSimulation.x = 0.0;
	AngleOfSimulation.y = 1.0;
	AngleOfSimulation.z = 0.0;
	
	BaseMuscleContractionDurationAdjustmentMultiplier = 1.0;
	BaseMuscleRechargeDurationAdjustmentMultiplier = 1.0;
	BaseMuscleConductionVelocityAdjustmentMultiplier = 1.0;

	DrawTimer = 0; 
	RecenterCount = 0;
	RunTime = 0.0;
	Pause = 1;
	MovieOn = 0;
	DrawNodesFlag = 0;
	DrawFrontHalfFlag = 0;
	
	AblateOnOff = 0;
	EctopicBeatOnOff = 0;
	AdjustMuscleOnOff = 0;
	FindNodeOnOff = 0;
	EctopicSingleOnOff = 0;
	MouseFunctionOnOff = 0;
	ViewFlag = 1;
	MovieFlag = 0;
	HitMultiplier = 0.03;
	MouseZ = RadiusOfAtria;
	ScrollSpeedToggle = 1;
	ScrollSpeed = 1.0;
	
	centerObject();
	setView(5);
	
	cudaMemcpy( MuscleGPU, Muscle, NumberOfMuscles*sizeof(muscleAtributesStructure), cudaMemcpyHostToDevice );
	errorCheck("cudaMemcpy Muscle up");
	cudaMemcpy( NodeGPU, Node, NumberOfNodes*sizeof(nodeAtributesStructure), cudaMemcpyHostToDevice );
	errorCheck("cudaMemcpy Node up");
	cudaMemcpy( ConnectingMusclesGPU, ConnectingMuscles, NumberOfNodes*LinksPerNode*sizeof(int), cudaMemcpyHostToDevice );
	errorCheck("cudaMemcpy ConnectingMuscles up");
	cudaMemcpy( EctopicEventsGPU, EctopicEvents, MaxNumberOfperiodicEctopicEvents*sizeof(ectopicEventStructure), cudaMemcpyHostToDevice );
	errorCheck("cudaMemcpy EctopicEvents up");
	
	//printf("\n\n The Particle Modeling Group hopes you enjoy your interactive simulation.");
	//printf("\n The simulation is paused type r to start the simulation and h for a help menu.");
	//printf("\n");
	
	printf("\n");
	terminalPrint();
}

int main(int argc, char** argv)
{
	setup();
	
	XWindowSize = 1000;
	YWindowSize = 1000; 

	// Clip plains
	Near = 0.2;
	Far = 80.0*RadiusOfAtria;

	//Direction here your eye is located location
	EyeX = 0.0*RadiusOfAtria;
	EyeY = 0.0*RadiusOfAtria;
	EyeZ = 2.0*RadiusOfAtria;

	//Where you are looking
	CenterX = 0.0;
	CenterY = 0.0;
	CenterZ = 0.0;

	//Up vector for viewing
	UpX = 0.0;
	UpY = 1.0;
	UpZ = 0.0;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(5,5);
	Window = glutCreateWindow("SVT");
	
	gluLookAt(EyeX, EyeY, EyeZ, CenterX, CenterY, CenterZ, UpX, UpY, UpZ);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, Near, Far);
	glMatrixMode(GL_MODELVIEW);
	glClearColor(BackGroundRed, BackGroundGreen, BackGroundBlue, 0.0);
	
	//GLfloat light_position[] = {EyeX, EyeY, EyeZ, 0.0};
	GLfloat light_position[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	
	//glutMouseFunc(mouseWheelCallback);
	//glutMouseWheelFunc(mouseWheelCallback);
	//glutMotionFunc(mouseMotionCallback);
    	glutPassiveMotionFunc(mousePassiveMotionCallback);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mymouse);
	glutKeyboardFunc(KeyPressed);
	glutIdleFunc(idle);
	glutSetCursor(GLUT_CURSOR_DESTROY);
	glEnable(GL_DEPTH_TEST);
	
	glutMainLoop();
	return 0;
}
