void Display(void);
void idle();
void reshape(int, int);
void mouseWheelCallback(int, int, int, int);
//void mouseMotionCallback(int, int);
void mousePassiveMotionCallback(int, int);
void orthoganialView();
void fulstrumView();
void mouseFunctionsOff();
void mouseAblateMode();
void mouseEctopicBeatMode();
void mouseEctopicEventMode();
void mouseAdjustMusclesMode();
void mouseIdentifyNodeMode();
int setMouseMuscleAttributes();
void setMouseMuscleContractionDuration();
void setMouseMuscleRechargeDuration();
void setMouseMuscleContractionVelocity();
void setEctopicBeat(int, int);
void getEctopicBeatPeriod(int);
void getEctopicBeatOffset(int);
void movieOn();
void movieOff();
void screenShot();
void saveSettings();
void helpMenu();
void KeyPressed(unsigned char, int, int);
void mymouse(int, int, int, int);

void clearStdin();

void Display(void)
{
	drawPicture();
}

void idle()
{
	n_body(Dt);
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);
}

void orthoganialView()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-RadiusOfAtria, RadiusOfAtria, -RadiusOfAtria, RadiusOfAtria, Near, Far);
	glMatrixMode(GL_MODELVIEW);
	ViewFlag = 0;
	drawPicture();
}

void fulstrumView()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, Near, Far);
	glMatrixMode(GL_MODELVIEW);
	ViewFlag = 1;
	drawPicture();
}

void mouseFunctionsOff()
{
	Pause = 1;
	AblateOnOff = 0;
	EctopicBeatOnOff = 0;
	EctopicSingleOnOff = 0;
	AdjustMuscleOnOff = 0;
	FindNodeOnOff = 0;
	MouseFunctionOnOff = 0;
	terminalPrint();
	glutSetCursor(GLUT_CURSOR_DESTROY);
	drawPicture();
}

void mouseAblateMode()
{
	Pause = 1;
	AblateOnOff = 1;
	EctopicBeatOnOff = 0;
	EctopicSingleOnOff = 0;
	AdjustMuscleOnOff = 0;
	FindNodeOnOff = 0;
	MouseFunctionOnOff = 1;
	glutSetCursor(GLUT_CURSOR_NONE);
	//orthoganialView();
	terminalPrint();
	drawPicture();
}

void mouseEctopicBeatMode()
{
	Pause = 1;
	AblateOnOff = 0;
	EctopicBeatOnOff = 1;
	EctopicSingleOnOff = 0;
	AdjustMuscleOnOff = 0;
	FindNodeOnOff = 0;
	MouseFunctionOnOff = 1;
	Pause = 1;
	glutSetCursor(GLUT_CURSOR_NONE);
	//orthoganialView();
	terminalPrint();
	drawPicture();
	system("clear");
	printf("\n You are in create ectopic beat mode.");
	printf("\n\n Use the mouse to select a node.");
	printf("\n");
}

void mouseEctopicEventMode()
{
	Pause = 1;
	AblateOnOff = 0;
	EctopicBeatOnOff = 0;
	EctopicSingleOnOff = 1;
	AdjustMuscleOnOff = 0;
	FindNodeOnOff = 0;
	MouseFunctionOnOff = 1;
	glutSetCursor(GLUT_CURSOR_NONE);
	//orthoganialView();
	drawPicture();
	terminalPrint();
}

void mouseAdjustMusclesMode()
{
	int returnCode = 0;
	Pause = 1;
	AblateOnOff = 0;
	EctopicBeatOnOff = 0;
	EctopicSingleOnOff = 0;
	AdjustMuscleOnOff = 1;
	FindNodeOnOff = 0;
	MouseFunctionOnOff = 1;
	glutSetCursor(GLUT_CURSOR_NONE);
	//orthoganialView();
	drawPicture();
	
	returnCode = setMouseMuscleAttributes();
	
	if(returnCode == 1)
	{
		terminalPrint();
	}
}

void mouseIdentifyNodeMode()
{
	Pause = 1;
	AblateOnOff = 0;
	EctopicBeatOnOff = 0;
	EctopicSingleOnOff = 0;
	AdjustMuscleOnOff = 0;
	FindNodeOnOff = 1;
	MouseFunctionOnOff = 1;
	glutSetCursor(GLUT_CURSOR_NONE);
	//orthoganialView();
	drawPicture();
	terminalPrint();
}

int setMouseMuscleAttributes()
{
	setMouseMuscleContractionDuration();
	setMouseMuscleRechargeDuration();
	setMouseMuscleContractionVelocity();
	return(1);
}

void setMouseMuscleContractionDuration()
{
	system("clear");
	BaseMuscleContractionDurationAdjustmentMultiplier = -1.0;
	
	printf("\n\n Enter the contraction duration multiplier.");
	printf("\n A number greater than 1 will make it longer.");
	printf("\n A number between 0 and 1 will make it shorter.");
	printf("\n\n Contraction duration multiplier = ");
	fflush(stdin);
	scanf("%f", &BaseMuscleContractionDurationAdjustmentMultiplier);
	if(BaseMuscleContractionDurationAdjustmentMultiplier < 0)
	{
		system("clear");
		printf("\n You cannot adjust the the contraction duration by a negative number.");
		printf("\n Retry\n");
		setMouseMuscleContractionDuration();
	}
}

void setMouseMuscleRechargeDuration()
{
	system("clear");
	BaseMuscleRechargeDurationAdjustmentMultiplier = -1.0;
	
	printf("\n\n Enter the recharge duration multiplier.");
	printf("\n A number greater than 1 will make it longer.");
	printf("\n A number between 0 and 1 will make it shorter.");
	printf("\n\n Recharge duration multiplier = ");
	fflush(stdin);
	scanf("%f", &BaseMuscleRechargeDurationAdjustmentMultiplier);
	if(BaseMuscleRechargeDurationAdjustmentMultiplier < 0)
	{
		system("clear");
		printf("\n You cannot adjust the the recharge duration by a negative number.");
		printf("\n Retry\n");
		setMouseMuscleRechargeDuration();
	}
}

void setMouseMuscleContractionVelocity()
{
	system("clear");
	BaseMuscleConductionVelocityAdjustmentMultiplier = -1.0;
	
	printf("\n\n Enter conduction velocity multipier.");
	printf("\n A number between 0 and 1 will slow it down.");
	printf("\n A number bigger than 1 will speed it up.");
	printf("\n\n Conduction velocity multiplier = ");
	fflush(stdin);
	scanf("%f", &BaseMuscleConductionVelocityAdjustmentMultiplier);
	if(BaseMuscleConductionVelocityAdjustmentMultiplier <= 0)
	{
		system("clear");
		printf("\n You cannot adjust the the conduction velocity by a nonpositive number.");
		printf("\n Retry\n");
		setMouseMuscleContractionVelocity();
	}
}

void setEctopicBeat(int nodeId, int event)
{
	EctopicEvents[event].node = nodeId;
	if(Node[nodeId].ablatedYesNo != 1)
	{
		Node[nodeId].drawFlag = 1;
		Node[nodeId].color.x = 1.0;
		Node[nodeId].color.y = 1.0;
		Node[nodeId].color.z = 0.0;
	}
	drawPicture();
	
	getEctopicBeatPeriod(event);
	getEctopicBeatOffset(event);
	
	// We only let you set 1 ectopic beat at a time.
	EctopicBeatOnOff = 0;
	terminalPrint();
}

void clearStdin()
{
    int c;
    while ((c = getchar()) != '\n' && c != EOF)
    {
        /* discard characters */
    }
}

void getEctopicBeatPeriod(int event)
{
	fflush(stdin);
	system("clear");
	printf("\n The current driving beat Period = %f.", BeatPeriod);
	printf("\n Enter the period of your ectopic beat.");
	
	printf("\n\n Ectopic period = ");
	scanf("%f", &EctopicEvents[event].period);
	if(EctopicEvents[event].period <= 0)
	{
		system("clear");
		printf("\n You entered %f.", EctopicEvents[event].period);
		printf("\n You cannot have a beat period that is a nonpositive number.");
		printf("\n Retry\n");
		exit(0);
		getEctopicBeatPeriod(event);
	}
	clearStdin();
}

void getEctopicBeatOffset(int event)
{
	system("clear");
	printf("\n The current Time into the beat is %f.", EctopicEvents[0].time);
	printf("\n Enter the time offset of your ectopic event.");
	printf("\n This will allow you to time your ectopic beat with the driving beat.");
	printf("\n Zero will start the ectopic beat now.");
	printf("\n A positive number will delay the ectopic beat by that amount.");
	printf("\n\n Ectopic time delay = ");
	fflush(stdin);
	float timeDelay;
	scanf("%f", &timeDelay);
	if(timeDelay < 0)
	{
		system("clear");
		printf("\n You cannot have a time delay that is a negative number.");
		printf("\n Retry\n");
		getEctopicBeatOffset(event);
	}
	EctopicEvents[event].time = EctopicEvents[event].period - timeDelay;
}

/*
	Returns a timestamp in M-D-Y-H.M.S format.
*/
string getTimeStamp()
{
	// Want to get a time stamp string representing current date/time, so we have a
	// unique name for each video/screenshot taken.
	time_t t = time(0); 
	struct tm * now = localtime( & t );
	int month = now->tm_mon + 1, day = now->tm_mday, year = now->tm_year, 
				curTimeHour = now->tm_hour, curTimeMin = now->tm_min, curTimeSec = now->tm_sec;
	stringstream smonth, sday, syear, stimeHour, stimeMin, stimeSec;
	smonth << month;
	sday << day;
	syear << (year + 1900); // The computer starts counting from the year 1900, so 1900 is year 0. So we fix that.
	stimeHour << curTimeHour;
	stimeMin << curTimeMin;
	stimeSec << curTimeSec;
	string timeStamp;

	if (curTimeMin <= 9)	
		timeStamp = smonth.str() + "-" + sday.str() + "-" + syear.str() + '_' + stimeHour.str() + ".0" + stimeMin.str() + 
					"." + stimeSec.str();
	else			
		timeStamp = smonth.str() + "-" + sday.str() + '-' + syear.str() + "_" + stimeHour.str() + "." + stimeMin.str() +
					"." + stimeSec.str();
	return timeStamp;
}

void movieOn()
{
	string ts = getTimeStamp();
	ts.append(".mp4");

	// Setting up the movie buffer.
	/*const char* cmd = "ffmpeg -loglevel quiet -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
		      "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output.mp4";*/

	string baseCommand = "ffmpeg -loglevel quiet -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
				"-c:v libx264rgb -threads 0 -preset fast -y -pix_fmt yuv420p -crf 0 -vf vflip ";

	string z = baseCommand + ts;

	const char *ccx = z.c_str();
	MovieFile = popen(ccx, "w");
	//Buffer = new int[XWindowSize*YWindowSize];
	Buffer = (int*)malloc(XWindowSize*YWindowSize*sizeof(int));
	MovieOn = 1;
}

void movieOff()
{
	if(MovieOn == 1) 
	{
		pclose(MovieFile);
	}
	free(Buffer);
	MovieOn = 0;
}

void screenShot()
{	
	int pauseFlag;
	FILE* ScreenShotFile;
	int* buffer;

	const char* cmd = "ffmpeg -loglevel quiet -framerate 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
				"-c:v libx264rgb -threads 0 -preset fast -y -crf 0 -vf vflip output1.mp4";
	//const char* cmd = "ffmpeg -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
	//              "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output1.mp4";
	ScreenShotFile = popen(cmd, "w");
	buffer = (int*)malloc(XWindowSize*YWindowSize*sizeof(int));
	
	if(Pause == 0) 
	{
		Pause = 1;
		pauseFlag = 0;
	}
	else
	{
		pauseFlag = 1;
	}
	
	for(int i =0; i < 1; i++)
	{
		drawPicture();
		glReadPixels(5, 5, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
		fwrite(buffer, sizeof(int)*XWindowSize*YWindowSize, 1, ScreenShotFile);
	}
	
	pclose(ScreenShotFile);
	free(buffer);

	string ts = getTimeStamp(); // Only storing in a separate variable for debugging purposes.
	string s = "ffmpeg -loglevel quiet -i output1.mp4 -qscale:v 1 -qmin 1 -qmax 1 " + ts + ".jpeg";
	// Convert back to a C-style string.
	const char *ccx = s.c_str();
	system(ccx);
	system("rm output1.mp4");
	printf("\nScreenshot Captured: \n");
	cout << "Saved as " << ts << ".jpeg" << endl;

	
	//system("ffmpeg -i output1.mp4 screenShot.jpeg");
	//system("rm output1.mp4");
	
	Pause = pauseFlag;
	//ffmpeg -i output1.mp4 output_%03d.jpeg
}

void saveSettings()
{
	cudaMemcpy( Node, NodeGPU, NumberOfNodes*sizeof(nodeAtributesStructure), cudaMemcpyDeviceToHost);
	errorCheck("cudaMemcpy Node down");
	cudaMemcpy( Muscle, MuscleGPU, NumberOfMuscles*sizeof(muscleAtributesStructure), cudaMemcpyDeviceToHost);
	errorCheck("cudaMemcpy Muscle down");
	cudaMemcpy( ConnectingMuscles, ConnectingMusclesGPU, NumberOfNodes*LinksPerNode*sizeof(int), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy ConnectingMuscles down");
	cudaMemcpy( EctopicEvents, EctopicEventsGPU, MaxNumberOfperiodicEctopicEvents*sizeof(ectopicEventStructure), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy EctopicEvents down");
	
	chdir("./PreviousRunsFile");
	   	
	//Create output file name to store run settings in.
	time_t t = time(0); 
	struct tm * now = localtime( & t );
	int month = now->tm_mon + 1, day = now->tm_mday, curTimeHour = now->tm_hour, curTimeMin = now->tm_min, curTimeSec = now->tm_sec;
	stringstream smonth, sday, stimeHour, stimeMin, stimeSec;
	smonth << month;
	sday << day;
	stimeHour << curTimeHour;
	stimeMin << curTimeMin;
	stimeSec << curTimeSec;
	string monthday;
	if(curTimeMin <= 9)
		if(curTimeSec <= 9) monthday = smonth.str() + "-" + sday.str() + "-" + stimeHour.str() + ":0" + stimeMin.str() + ":0" + stimeSec.str();
		else monthday = smonth.str() + "-" + sday.str() + "-" + stimeHour.str() + ":0" + stimeMin.str() + ":" + stimeSec.str();
	else monthday = smonth.str() + "-" + sday.str() + "-" + stimeHour.str() + ":" + stimeMin.str() + ":" + stimeSec.str();
	string timeStamp = "Run:" + monthday;
	const char *diretoryName = timeStamp.c_str();
	
	if(mkdir(diretoryName, 0777) == 0)
	{
		printf("\n Directory '%s' created successfully.\n", diretoryName);
	}
	else
	{
		printf("\n Error creating directory '%s'.\n", diretoryName);
	}
	
	chdir(diretoryName);
	
	// Copying all the nodes and muscle (with their properties) into this folder in the file named run.
	FILE *settingFile;
  	settingFile = fopen("run", "wb");
  	fwrite(&PulsePointNode, sizeof(int), 1, settingFile);
  	fwrite(&UpNode, sizeof(int), 1, settingFile);
  	fwrite(&FrontNode, sizeof(int), 1, settingFile);
  	fwrite(&NumberOfNodes, sizeof(int), 1, settingFile);
  	fwrite(&NumberOfMuscles, sizeof(int), 1, settingFile);
  	fwrite(&LinksPerNode, sizeof(int), 1, settingFile);
  	fwrite(&MaxNumberOfperiodicEctopicEvents, sizeof(int), 1, settingFile);
  	fwrite(Node, sizeof(nodeAtributesStructure), NumberOfNodes, settingFile);
  	fwrite(Muscle, sizeof(muscleAtributesStructure), NumberOfMuscles, settingFile);
  	fwrite(ConnectingMuscles, sizeof(int), NumberOfNodes*LinksPerNode, settingFile);
  	fwrite(EctopicEvents, sizeof(ectopicEventStructure), MaxNumberOfperiodicEctopicEvents, settingFile);
	fclose(settingFile);
	
	//Copying the simulationSetup file into this directory so you will know how it was initally setup.
	FILE *fileIn;
	FILE *fileOut;
	long sizeOfFile;
  	char *buffer;
	fileIn = fopen("../../simulationSetup", "rb");
	if(fileIn == NULL)
	{
		printf("\n\n The simulationSetup file does not exist\n\n");
		exit(0);
	}
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
	fileOut = fopen("simulationSetup", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);
	
	// Making a readMe file to put any infomation about why you are saving this run.
	system("gedit readMe");
	
	// Moving back to the SVT directory.
	chdir("../");
}

void helpMenu()
{
	system("clear");
	//Pause = 1;
	printf("\n The simulation is paused.");
	printf("\n");
	printf("\n h: Help");
	printf("\n q: Quit");
	printf("\n r: Run/Pause (Toggle)");
	printf("\n g: View front half only/View full image (Toggle)");
	printf("\n n: Nodes off/half/full (Toggle)");
	printf("\n v: Orthogonal/Frustum projection (Toggle)");
	printf("\n");
	printf("\n m: Movie on/Movie off (Toggle)");
	printf("\n S: Screenshot");
	printf("\n");
	printf("\n Views: 7 8 9 | LL  SUP RL" );
	printf("\n Views: 4 5 6 | PA  INF Ref" );
	printf("\n Views: 1 2 3 | LOA AP  ROA" );
	printf("\n");
	printf("\n c: Recenter image");
	printf("\n w: Counterclockwise rotation x-axis");
	printf("\n s: Clockwise rotation x-axis");
	printf("\n d: Counterclockwise rotation y-axis");
	printf("\n a: Clockwise rotation y-axis");
	printf("\n z: Counterclockwise rotation z-axis");
	printf("\n Z: Clockwise rotation z-axis");
	printf("\n e: Zoom in");
	printf("\n E: Zoom out");
	printf("\n");
	printf("\n [ or ]: Increases/Decrease the selection area of the mouse");
	printf("\n shift 0: Turns off all mouse action.");
	printf("\n shift 1: Turns on ablating. Left mouse ablate node. Right mouse undo ablation.");
	printf("\n shift 2: Turns on ectopic beat. Left mouse set node as an ectopic beat location.");
	printf("\n Note this action will prompt you to enter the");
	printf("\n beat period and time offset in the terminal.");
	printf("\n shift 3: Turns on one ectopic trigger.");
	printf("\n Left mouse will trigger that node to start a single pulse at that location.");
	printf("\n shift 4: Turns on muscle adjustments. Left mouse set node muscles adjustments.");
	printf("\n Note this action will prompt you to entire the ");
	printf("\n contraction, recharge, and action potential adjustment multiplier in the terminal.");
	printf("\n shift 5: Turns on find node. Left mouse displays the Id of the node in the terminal.");
	printf("\n");
	printf("\n k: Save your current muscle attributes (note: previous run files are ignored by git)");
	printf("\n ?: Find the up and front node at current view.");
	printf("\n");
}

void KeyPressed(unsigned char key, int x, int y)
{
	float dAngle = 0.01;
	float zoom = 0.01*RadiusOfAtria;
	float temp;
	float4 lookVector;
	float d;
	float4 centerOfMass;
	
	copyNodesMusclesFromGPU();
	
	lookVector.x = CenterX - EyeX;
	lookVector.y = CenterY - EyeY;
	lookVector.z = CenterZ - EyeZ;
	d = sqrt(lookVector.x*lookVector.x + lookVector.y*lookVector.y + lookVector.z*lookVector.z);
	if(d < 0.00001)
	{
		printf("\n lookVector is too small\n");
		printf("\n Good Bye\n");
		exit(0);
	}
	else
	{
		lookVector.x /= d;
		lookVector.y /= d;
		lookVector.z /= d;
	}
	
	centerOfMass = findCenterOfMass();
	
	if(key == 'h')  // Help menu
	{
		helpMenu();
	}
	
	if(key == 'q') // quit
	{
		glutDestroyWindow(Window);
		printf("\n Good Bye\n");
		exit(0);
	}
	
	if(key == 'r')  // Run toggle
	{
		if(Pause == 0) Pause = 1;
		else Pause = 0;
		terminalPrint();
	}
	if(key == 'n')  // Draw nodes toggle
	{
		if(DrawNodesFlag == 0) DrawNodesFlag = 1;
		else if(DrawNodesFlag == 1) DrawNodesFlag = 2;
		else DrawNodesFlag = 0;
		drawPicture();
	terminalPrint();
	}
	if(key == 'g')  // Draw full or front half toggle
	{
		if(DrawFrontHalfFlag == 0) DrawFrontHalfFlag = 1;
		else DrawFrontHalfFlag = 0;
		drawPicture();
		terminalPrint();
	}
	if(key == 'B')  // Raising the beat period
	{
		EctopicEvents[0].period += 10;
		//EctopicEvents[0].time = EctopicEvents[0].period;
		cudaMemcpy( EctopicEventsGPU, EctopicEvents, MaxNumberOfperiodicEctopicEvents*sizeof(ectopicEventStructure), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy EctopicEvents up");
		terminalPrint();
	}
	if(key == 'b')  // Lowering the beat period
	{
		EctopicEvents[0].period -= 10;
		if(EctopicEvents[0].period < 0) 
		{
			EctopicEvents[0].period = 0;  // You don't want the beat to negative
		}
		cudaMemcpy( EctopicEventsGPU, EctopicEvents, MaxNumberOfperiodicEctopicEvents*sizeof(ectopicEventStructure), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy EctopicEvents up");
		//EctopicEvents[0].time = EctopicEvents[0].period;
		terminalPrint();
	}
	if(key == 'v') // Orthoganal view
	{
		if(ViewFlag == 0) 
		{
			ViewFlag = 1;
			fulstrumView();
		}
		else 
		{
			ViewFlag = 0;
			orthoganialView();
		}
		drawPicture();
		terminalPrint();
	}
	
	if(key == 'm')  // Movie on
	{
		if(MovieFlag == 0) 
		{
			MovieFlag = 1;
			movieOn();
		}
		else 
		{
			MovieFlag = 0;
			movieOff();
		}
		terminalPrint();
	}
	
	if(key == 'S')  // Screenshot
	{	
		screenShot();
		terminalPrint();
	}
	
	if(key == '0')
	{
		setView(0);
		drawPicture();
		terminalPrint();
	}
	if(key == '1')
	{
		setView(1);
		drawPicture();
		terminalPrint();
	}
	if(key == '2')
	{
		setView(2);
		drawPicture();
		terminalPrint();
	}
	if(key == '3')
	{
		setView(3);
		drawPicture();
		terminalPrint();
	}
	if(key == '4')
	{
		setView(4);
		drawPicture();
		terminalPrint();
	}
	if(key == '5')
	{
		setView(5);
		drawPicture();
		terminalPrint();
	}
	if(key == '6')
	{
		setView(6);
		drawPicture();
		terminalPrint();
	}
	if(key == '7')
	{
		setView(7);
		drawPicture();
		terminalPrint();
	}
	if(key == '8')
	{
		setView(8);
		drawPicture();
		terminalPrint();
	}
	if(key == '9')
	{
		setView(9);
		drawPicture();
		terminalPrint();
	}
	if(key == '?') // Finding front and top reference nodes.
	{
		float maxZ = -10000.0;
		float maxY = -10000.0;
		int indexZ = -1;
		int indexY = -1;
		
		for(int i = 0; i < NumberOfNodes; i++)
		{
			if(maxZ < Node[i].position.z) 
			{
				maxZ = Node[i].position.z;
				indexZ = i;
			}
			
			if(maxY < Node[i].position.y) 
			{
				maxY = Node[i].position.y;
				indexY = i;
			}
		}
		
		Node[indexZ].color.x = 0.0;
		Node[indexZ].color.y = 0.0;
		Node[indexZ].color.z = 1.0;
		
		Node[indexY].color.x = 1.0;
		Node[indexY].color.y = 0.0;
		Node[indexY].color.z = 1.0;
		
		system("clear");
		printf("\n Front node index = %d\n", indexZ);
		printf("\n Top node index   = %d\n", indexY);
		
		drawPicture();
	}
	if(key == 'w')  // Rotate counterclockwise on the x-axis
	{
		for(int i = 0; i < NumberOfNodes; i++)
		{
			Node[i].position.x -= centerOfMass.x;
			Node[i].position.y -= centerOfMass.y;
			Node[i].position.z -= centerOfMass.z;
			temp = cos(dAngle)*Node[i].position.y - sin(dAngle)*Node[i].position.z;
			Node[i].position.z  = sin(dAngle)*Node[i].position.y + cos(dAngle)*Node[i].position.z;
			Node[i].position.y  = temp;
			Node[i].position.x += centerOfMass.x;
			Node[i].position.y += centerOfMass.y;
			Node[i].position.z += centerOfMass.z;
		}
		drawPicture();
		AngleOfSimulation.x += dAngle;
	}
	if(key == 's')  // Rotate clockwise on the x-axis
	{
		for(int i = 0; i < NumberOfNodes; i++)
		{
			Node[i].position.x -= centerOfMass.x;
			Node[i].position.y -= centerOfMass.y;
			Node[i].position.z -= centerOfMass.z;
			temp = cos(-dAngle)*Node[i].position.y - sin(-dAngle)*Node[i].position.z;
			Node[i].position.z  = sin(-dAngle)*Node[i].position.y + cos(-dAngle)*Node[i].position.z;
			Node[i].position.y  = temp; 
			Node[i].position.x += centerOfMass.x;
			Node[i].position.y += centerOfMass.y;
			Node[i].position.z += centerOfMass.z;
		}
		drawPicture();
		AngleOfSimulation.x -= dAngle;
	}
	if(key == 'd')  // Rotate counterclockwise on the y-axis
	{
		for(int i = 0; i < NumberOfNodes; i++)
		{
			Node[i].position.x -= centerOfMass.x;
			Node[i].position.y -= centerOfMass.y;
			Node[i].position.z -= centerOfMass.z;
			temp =  cos(-dAngle)*Node[i].position.x + sin(-dAngle)*Node[i].position.z;
			Node[i].position.z  = -sin(-dAngle)*Node[i].position.x + cos(-dAngle)*Node[i].position.z;
			Node[i].position.x  = temp;
			Node[i].position.x += centerOfMass.x;
			Node[i].position.y += centerOfMass.y;
			Node[i].position.z += centerOfMass.z;
		}
		drawPicture();
		AngleOfSimulation.y -= dAngle;
	}
	if(key == 'a')  // Rotate clockwise on the y-axis
	{
		for(int i = 0; i < NumberOfNodes; i++)
		{
			Node[i].position.x -= centerOfMass.x;
			Node[i].position.y -= centerOfMass.y;
			Node[i].position.z -= centerOfMass.z;
			temp = cos(dAngle)*Node[i].position.x + sin(dAngle)*Node[i].position.z;
			Node[i].position.z  = -sin(dAngle)*Node[i].position.x + cos(dAngle)*Node[i].position.z;
			Node[i].position.x  = temp;
			Node[i].position.x += centerOfMass.x;
			Node[i].position.y += centerOfMass.y;
			Node[i].position.z += centerOfMass.z;
		}
		drawPicture();
		AngleOfSimulation.y += dAngle;
	}
	if(key == 'z')  // Rotate counterclockwise on the z-axis
	{
		for(int i = 0; i < NumberOfNodes; i++)
		{
			Node[i].position.x -= centerOfMass.x;
			Node[i].position.y -= centerOfMass.y;
			Node[i].position.z -= centerOfMass.z;
			temp = cos(dAngle)*Node[i].position.x - sin(dAngle)*Node[i].position.y;
			Node[i].position.y  = sin(dAngle)*Node[i].position.x + cos(dAngle)*Node[i].position.y;
			Node[i].position.x  = temp;
			Node[i].position.x += centerOfMass.x;
			Node[i].position.y += centerOfMass.y;
			Node[i].position.z += centerOfMass.z;
		}
		drawPicture();
		AngleOfSimulation.z += dAngle;
	}
	if(key == 'Z')  // Rotate clockwise on the z-axis
	{
		for(int i = 0; i < NumberOfNodes; i++)
		{
			Node[i].position.x -= centerOfMass.x;
			Node[i].position.y -= centerOfMass.y;
			Node[i].position.z -= centerOfMass.z;
			temp = cos(-dAngle)*Node[i].position.x - sin(-dAngle)*Node[i].position.y;
			Node[i].position.y  = sin(-dAngle)*Node[i].position.x + cos(-dAngle)*Node[i].position.y;
			Node[i].position.x  = temp;
			Node[i].position.x += centerOfMass.x;
			Node[i].position.y += centerOfMass.y;
			Node[i].position.z += centerOfMass.z;
		}
		drawPicture();
		AngleOfSimulation.z -= dAngle;
	}
	if(key == 'e')  // Zoom in
	{
		for(int i = 0; i < NumberOfNodes; i++)
		{
			Node[i].position.x -= zoom*lookVector.x;
			Node[i].position.y -= zoom*lookVector.y;
			Node[i].position.z -= zoom*lookVector.z;
		}
		CenterOfSimulation.x -= zoom*lookVector.x;
		CenterOfSimulation.y -= zoom*lookVector.y;
		CenterOfSimulation.z -= zoom*lookVector.z;
		drawPicture();
	}
	if(key == 'E')  // Zoom out
	{
		for(int i = 0; i < NumberOfNodes; i++)
		{
			Node[i].position.x += zoom*lookVector.x;
			Node[i].position.y += zoom*lookVector.y;
			Node[i].position.z += zoom*lookVector.z;
		}
		CenterOfSimulation.x += zoom*lookVector.x;
		CenterOfSimulation.y += zoom*lookVector.y;
		CenterOfSimulation.z += zoom*lookVector.z;
		drawPicture();
	}
	
	if(key == ')')  // All mouse functions are off (shift 0)
	{
		mouseFunctionsOff();
		MouseFunctionOnOff = 0;
	}
	if(key == '!')  // Ablate is on (shift 1)
	{
		mouseAblateMode();
		MouseFunctionOnOff = 1;
	}
	if(key == '@')  // Ectopic beat is on (shift 2)
	{
		mouseEctopicBeatMode();
		MouseFunctionOnOff = 1;
	}
	if(key == '#')  // You are in ectopic single trigger mode. (shift 3)
	{
		mouseEctopicEventMode();
		MouseFunctionOnOff = 1;
	}
	if(key == '$') // muscle adjustment is on (shift 4)
	{
		mouseAdjustMusclesMode();
		MouseFunctionOnOff = 1;
	}
	if(key == '^')  // Find node is on (shift 5)
	{
		mouseIdentifyNodeMode();
		MouseFunctionOnOff = 1;
	}
	
	if(key == ']')  
	{
		HitMultiplier += 0.005;
		terminalPrint();
		//printf("\n Your selection area = %f times the radius of atrium. \n", HitMultiplier);
	}
	if(key == '[')
	{
		HitMultiplier -= 0.005;
		if(HitMultiplier < 0.0) HitMultiplier = 0.0;
		terminalPrint();
		//printf("\n Your selection area = %f times the radius of atrium. \n", HitMultiplier);
	}
	
	if(key == 'c')  // Recenter the simulation
	{
		centerObject();
		drawPicture();
	}
	
	if(key == 'k')  // Save your current setting so you can start with this run in the future.
	{
		saveSettings();
	}
	
	copyNodesMusclesToGPU();
}


void mousePassiveMotionCallback(int x, int y) 
{
	// This function is called when the mouse moves without any button pressed
	// x and y are the current mouse coordinates
	// Use these coordinates as needed
	
	// x and y come in as 0 to XWindowSize and 0 to YWindowSize. This traveslates them to -1 to 1 and -1 to 1.

	MouseX = ( 2.0*x/XWindowSize - 1.0)*RadiusOfAtria;
	MouseY = (-2.0*y/YWindowSize + 1.0)*RadiusOfAtria;
	//drawPicture();

	//printf("\n MouseX = %f\n", MouseX);
}

void mymouse(int button, int state, int x, int y)
{	
	float dx, dy, dz;
	float hit;
	int muscleId;
	int event;
	
	if(state == GLUT_DOWN)
	{
		copyNodesMusclesFromGPU();
		hit = HitMultiplier*RadiusOfAtria;
		
		if(button == GLUT_LEFT_BUTTON)
		{	
			for(int i = 0; i < NumberOfNodes; i++)
			{
				dx = MouseX - Node[i].position.x;
				dy = MouseY - Node[i].position.y;
				dz = MouseZ - Node[i].position.z;
				
				if(sqrt(dx*dx + dy*dy + dz*dz) < hit)
				{
					if(AblateOnOff == 1)
					{
						Node[i].ablatedYesNo = 1;
						Node[i].drawFlag = 1;
						Node[i].color.x = 1.0;
						Node[i].color.y = 1.0;
						Node[i].color.z = 1.0;
					}
					if(EctopicBeatOnOff == 1)
					{
						cudaMemcpy( EctopicEvents, EctopicEventsGPU, MaxNumberOfperiodicEctopicEvents*sizeof(ectopicEventStructure), cudaMemcpyDeviceToHost);
						errorCheck("cudaMemcpy EctopicEvents down");
						
						Pause = 1;
						
						event = 1; // 0 is the sinus beat.
						while(EctopicEvents[event].node != -1 && event < MaxNumberOfperiodicEctopicEvents)
						{
							event++;
						}
						if(event < MaxNumberOfperiodicEctopicEvents)
						{
							setEctopicBeat(i, event);
						}
						else
						{
							printf("\n You are past the number of ectopic signals you can generate.\n");
						}
					
						cudaMemcpy( EctopicEventsGPU, EctopicEvents, MaxNumberOfperiodicEctopicEvents*sizeof(ectopicEventStructure), cudaMemcpyHostToDevice );
						errorCheck("cudaMemcpy EctopicEvents up");
					}
					if(AdjustMuscleOnOff == 1)
					{
						for(int j = 0; j < LinksPerNode; j++)
						{
							muscleId = ConnectingMuscles[i*LinksPerNode + j];
							if(muscleId != -1)
							{
								Muscle[muscleId].contractionDuration = BaseMuscleContractionDuration*BaseMuscleContractionDurationAdjustmentMultiplier;
								Muscle[muscleId].rechargeDuration = BaseMuscleRechargeDuration*BaseMuscleRechargeDurationAdjustmentMultiplier;
								Muscle[muscleId].conductionVelocity = BaseMuscleConductionVelocity*BaseMuscleConductionVelocityAdjustmentMultiplier;
								Muscle[muscleId].conductionDuration = Muscle[muscleId].naturalLength/Muscle[muscleId].conductionVelocity;
								Muscle[muscleId].color.x = 1.0;
								Muscle[muscleId].color.y = 0.0;
								Muscle[muscleId].color.z = 1.0;
								Muscle[muscleId].color.w = 0.0;
								
								if((Muscle[muscleId].contractionDuration + Muscle[muscleId].rechargeDuration) < Muscle[muscleId].conductionDuration)
								{
								 	printf("\n Conduction duration is shorter than the (contraction plus recharge) duration in muscle number %d", muscleId);
								 	printf("\n Muscle %d will be killed \n", muscleId);
								 	Muscle[muscleId].dead = 1;
								 	Muscle[muscleId].color.x = DeadColor.x;
									Muscle[muscleId].color.y = DeadColor.y;
									Muscle[muscleId].color.z = DeadColor.z;
									Muscle[muscleId].color.w = 1.0;
								} 
							}
						}
						
						Node[i].drawFlag = 1;
						if(Node[i].ablatedYesNo != 1) // If it is not ablated color it.
						{
							Node[i].color.x = 0.0;
							Node[i].color.y = 0.0;
							Node[i].color.z = 1.0;
						}
					}
					if(EctopicSingleOnOff == 1)
					{
						//Turning on the muscles associated with node i.
						if(Node[i].ablatedYesNo != 1) // If the node is ablated just return.
						{
							int nodeNumber;
							int muscleNumber;
							for(int j = 0; j < LinksPerNode; j++)
							{
								if(NumberOfNodes*LinksPerNode <= (i*LinksPerNode + j))
								{
									printf("\nTSU Error: number of ConnectingMuscles is out of bounds\n");
									exit(0);
								}
								muscleNumber = ConnectingMuscles[i*LinksPerNode + j];
								// Making sure there is a connection then making sure you do not turn on the muscle you just left.
								if((muscleNumber != -1)) 
								{
									if(i != Muscle[muscleNumber].apNode)
									{
										nodeNumber = Muscle[muscleNumber].nodeA;
										if(nodeNumber == i) nodeNumber = Muscle[muscleNumber].nodeB;
										if(Muscle[muscleNumber].onOff == 0)
										{
											Muscle[muscleNumber].apNode = i;  //This is the node where the AP wave started.
											Muscle[muscleNumber].onOff = 1;
											Muscle[muscleNumber].timer = 0.0;
										}
									}
								}
							}
						}
					}
					if(FindNodeOnOff == 1)
					{
						Node[i].drawFlag = 1;
						Node[i].color.x = 1.0;
						Node[i].color.y = 0.0;
						Node[i].color.z = 1.0;
						printf("\n Node number = %d", i);
					}
				}
			}
		}
		else if(button == GLUT_RIGHT_BUTTON) // Right Mouse button down
		{
			for(int i = 0; i < NumberOfNodes; i++)
			{
				dx = MouseX - Node[i].position.x;
				dy = MouseY - Node[i].position.y;
				dz = MouseZ - Node[i].position.z;
				if(sqrt(dx*dx + dy*dy + dz*dz) < hit)
				{
					if(AblateOnOff == 1)
					{
						Node[i].ablatedYesNo = 0;
						Node[i].drawFlag = 0;
						Node[i].color.x = 0.0;
						Node[i].color.y = 1.0;
						Node[i].color.z = 0.0;
					}
					if(AdjustMuscleOnOff == 1)
					{
						for(int j = 0; j < LinksPerNode; j++)
						{
							muscleId = ConnectingMuscles[i*LinksPerNode + j];
							if(muscleId != -1)
							{
								Muscle[muscleId].contractionDuration = BaseMuscleContractionDuration;
								Muscle[muscleId].rechargeDuration = BaseMuscleRechargeDuration;
								Muscle[muscleId].conductionVelocity = BaseMuscleConductionVelocity;
								Muscle[muscleId].conductionDuration = Muscle[muscleId].naturalLength/Muscle[muscleId].conductionVelocity;
								Muscle[muscleId].color.x = 0.0;
								Muscle[muscleId].color.y = 1.0;
								Muscle[muscleId].color.z = 0.0;
								Muscle[muscleId].color.w = 0.0;
								
								// Turning the muscle back on if it was dead.
								Muscle[muscleId].dead = 0;
								
								// Checking to see if the muscle needs to be killed.
								if((Muscle[muscleId].contractionDuration + Muscle[muscleId].rechargeDuration) < Muscle[muscleId].conductionDuration)
								{
								 	printf("\n Conduction duration is shorter than the (contraction plus recharge) duration in muscle number %d", muscleId);
								 	printf("\n Muscle %d will be killed \n", muscleId);
								 	Muscle[muscleId].dead = 1;
								 	Muscle[muscleId].color.x = DeadColor.x;
									Muscle[muscleId].color.y = DeadColor.y;
									Muscle[muscleId].color.z = DeadColor.z;
									Muscle[muscleId].color.w = 1.0;
								} 
							}
						}
						
						Node[i].drawFlag = 1;
						if(Node[i].ablatedYesNo != 1) // If it is not ablated color it.
						{
							Node[i].color.x = 0.0;
							Node[i].color.y = 1.0;
							Node[i].color.z = 0.0;
						}
					}
				}
			}
		}
		else if(button == GLUT_MIDDLE_BUTTON)
		{
			if(ScrollSpeedToggle == 0)
			{
				ScrollSpeedToggle = 1;
				ScrollSpeed = 1.0;
				printf("\n speed = %f\n", ScrollSpeed);
			}
			else
			{
				ScrollSpeedToggle = 0;
				ScrollSpeed = 0.1;
				printf("\n speed = %f\n", ScrollSpeed);
			}
			
		}
		drawPicture();
		copyNodesMusclesToGPU();
		//printf("\nSNx = %f SNy = %f SNz = %f\n", NodePosition[0].x, NodePosition[0].y, NodePosition[0].z);
	}
	
	if(state == 0)
	{
		if(button == 3) //Scroll up
		{
			MouseZ -= ScrollSpeed;
		}
		else if(button == 4) //Scroll down
		{
			MouseZ += ScrollSpeed;
		}
		//printf("MouseZ = %f\n", MouseZ);
		drawPicture();
	}
}

