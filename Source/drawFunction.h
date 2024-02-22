void rotateXAxis(float angle)
{
	float temp;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		temp = cos(angle)*Node[i].position.y - sin(angle)*Node[i].position.z;
		Node[i].position.z  = sin(angle)*Node[i].position.y + cos(angle)*Node[i].position.z;
		Node[i].position.y  = temp;
	}
	AngleOfSimulation.x += angle;
}

void rotateYAxis(float angle)
{
	float temp;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		temp =  cos(-angle)*Node[i].position.x + sin(-angle)*Node[i].position.z;
		Node[i].position.z  = -sin(-angle)*Node[i].position.x + cos(-angle)*Node[i].position.z;
		Node[i].position.x  = temp;
	}
	AngleOfSimulation.y += angle;
}

void rotateZAxis(float angle)
{
	float temp;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		temp = cos(angle)*Node[i].position.x - sin(angle)*Node[i].position.y;
		Node[i].position.y  = sin(angle)*Node[i].position.x + cos(angle)*Node[i].position.y;
		Node[i].position.x  = temp;
	}
	AngleOfSimulation.z += angle;
}

void ReferenceView()
{
	// We set the reference view looking straight at the four
	// pulminary viens with a vien in each of the four quadrants 
	// of the x-y plane as semetric as you can make it with the 
	// valve down.
	
	float angle, temp;
	
	float4 centerOfMass = findCenterOfMass();
	for(int i = 0; i < NumberOfNodes; i++)
	{
		Node[i].position.x -= centerOfMass.x;
		Node[i].position.y -= centerOfMass.y;
		Node[i].position.z -= centerOfMass.z;
	}
		
	// Rotating until the up Node is on x-y plane above or below the positive x-axis.
	angle = atan(Node[UpNode].position.z/Node[UpNode].position.x);
	if(Node[UpNode].position.x < 0.0) angle -= PI;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		temp = cos(angle)*Node[i].position.x + sin(angle)*Node[i].position.z;
		Node[i].position.z  = -sin(angle)*Node[i].position.x + cos(angle)*Node[i].position.z;
		Node[i].position.x  = temp;
	}
	AngleOfSimulation.y += angle;
	
	// Rotating until up Node is on the positive y axis.
	angle = PI/2.0 - atan(Node[UpNode].position.y/Node[UpNode].position.x);
	for(int i = 0; i < NumberOfNodes; i++)
	{
		temp = cos(angle)*Node[i].position.x - sin(angle)*Node[i].position.y;
		Node[i].position.y  = sin(angle)*Node[i].position.x + cos(angle)*Node[i].position.y;
		Node[i].position.x  = temp;
	}
	AngleOfSimulation.z += angle;
	
	// Rotating until front Node is on the positive z axis.
	angle = atan(Node[FrontNode].position.z/Node[FrontNode].position.x) - PI/2.0;
	if(Node[FrontNode].position.x < 0.0) angle -= PI;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		temp = cos(angle)*Node[i].position.x + sin(angle)*Node[i].position.z;
		Node[i].position.z  = -sin(angle)*Node[i].position.x + cos(angle)*Node[i].position.z;
		Node[i].position.x  = temp;
	}
	AngleOfSimulation.y += angle;
}

void PAView()
{  
	// The heart does not set in the chest at a straight on angle.
	// Hense we need to adjust our view to what they use in the EP lab.
	// We first set the AP view because it is the closest to ours.
	float angle;
	
	angle = PI/6.0; // Rotate 30 degrees counterclockwise on the y-axis 
	rotateYAxis(angle);

	angle = PI/6.0; // Rotate 30 degrees counterclockwise on the z-axis
	rotateZAxis(angle);
}

void APView()
{ 
	// To get the AP view we just rotate the PA view 180 degrees on the y-axis
	float angle;
	
	PAView();
	angle = PI; // Rotate 180 degrees counterclockwise on the y-axis 
	rotateYAxis(angle);
}

void setView(int view)
{
	// Putting object into reference view because everything is based off of this.
	ReferenceView();
	
	if(view == 6)
	{
		strcpy(ViewName, "Ref");
	}
	if(view == 4)
	{
		PAView();
		strcpy(ViewName, "PA");
	}
	if(view == 2)
	{
		APView();
		strcpy(ViewName, "AP");
	}
	if(view == 3)
	{
		APView();
		rotateYAxis(-PI/6.0);
		strcpy(ViewName, "RAO");
	}
	if(view == 1)
	{
		APView();
		rotateYAxis(PI/3.0);
		strcpy(ViewName, "LAO");
	}
	if(view == 7)
	{
		APView();
		rotateYAxis(PI/2.0);
		strcpy(ViewName, "LL");
	}
	if(view == 9)
	{
		APView();
		rotateYAxis(-PI/2.0);
		strcpy(ViewName, "RL");
	}
	if(view == 8)
	{
		APView();
		rotateXAxis(PI/2.0);
		strcpy(ViewName, "SUP");
	}
	if(view == 5)
	{
		APView();
		rotateXAxis(-PI/2.0);
		strcpy(ViewName, "INF");
	}
	else
	{
		printf("\n Undefined view reverting back to Ref view.");
	}
}

void drawPicture()
{
	//int nodeNumber;
	int muscleNumber;
	int k;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	// Draw Nodes
	// Sinus node
	if(Pause == 0) glColor3d(0.0,1.0,0.0);
	else glColor3d(1.0,0.0,0.0);
	glPushMatrix();
	glTranslatef(Node[PulsePointNode].position.x, Node[PulsePointNode].position.y, Node[PulsePointNode].position.z);
	glutSolidSphere(0.03*RadiusOfAtria,20,20);
	glPopMatrix();
	
	// Drawing center node
	/*
	glColor3d(1.0,1.0,1.0);
	glPushMatrix();
	glTranslatef(CenterOfSimulation.x, CenterOfSimulation.y, CenterOfSimulation.z);
	glutSolidSphere(0.02*RadiusOfAtria,20,20);
	glPopMatrix();
	*/

	// Drawing other nodes
	if(DrawNodesFlag == 1 || DrawNodesFlag == 2)
	{
		for(int i = 1; i < NumberOfNodes; i++)
		{
			if(DrawFrontHalfFlag == 1 || DrawNodesFlag == 1)
			{
				if(CenterOfSimulation.z - 0.001 < Node[i].position.z)  // Only drawing the nodes in the front.
				{
					glColor3d(Node[i].color.x, Node[i].color.y, Node[i].color.z);
					glPushMatrix();
					glTranslatef(Node[i].position.x, Node[i].position.y, Node[i].position.z);
					glutSolidSphere(NodeRadiusAdjustment*RadiusOfAtria,20,20);
					glPopMatrix();
				}
			}
			else
			{
				glColor3d(Node[i].color.x, Node[i].color.y, Node[i].color.z);
				glPushMatrix();
				glTranslatef(Node[i].position.x, Node[i].position.y, Node[i].position.z);
				glutSolidSphere(NodeRadiusAdjustment*RadiusOfAtria,20,20);
				glPopMatrix();
			}	
		}
	}
	else
	{
		glPointSize(5.0);
		glBegin(GL_POINTS);
		 	for(int i = 1; i < NumberOfNodes; i++)
			{
				if(DrawFrontHalfFlag == 1)
				{
					if(CenterOfSimulation.z - 0.001 < Node[i].position.z)  // Only drawing the nodes in the front.
					{
						glColor3d(Node[i].color.x, Node[i].color.y, Node[i].color.z);
						if(Node[i].drawFlag == 1)
						{
							glVertex3f(Node[i].position.x, Node[i].position.y, Node[i].position.z);
						}
					}
				}
				else
				{
					glColor3d(Node[i].color.x, Node[i].color.y, Node[i].color.z);
					if(Node[i].drawFlag == 1)
					{
						glVertex3f(Node[i].position.x, Node[i].position.y, Node[i].position.z);
					}
				}
			}
		glEnd();
	}
	
	// Drawing muscles
	glLineWidth(LineWidth);
	for(int i = 0; i < NumberOfNodes; i++)
	{
		for(int j = 0; j < LinksPerNode; j++)
		{
			muscleNumber = ConnectingMuscles[i*LinksPerNode + j];
			if(muscleNumber != -1)
			{
				k = Muscle[muscleNumber].nodeA;
				if(k == i) 
				{
					k = Muscle[muscleNumber].nodeB;
				}
				
				if(DrawFrontHalfFlag == 1)
				{
					if(CenterOfSimulation.z - 0.001 < Node[i].position.z && CenterOfSimulation.z - 0.001 < Node[k].position.z)  // Only drawing the nodes in the front.
					{
						glColor3d(Muscle[muscleNumber].color.x, Muscle[muscleNumber].color.y, Muscle[muscleNumber].color.z);
						glBegin(GL_LINES);
							glVertex3f(Node[i].position.x, Node[i].position.y, Node[i].position.z);
							glVertex3f(Node[k].position.x, Node[k].position.y, Node[k].position.z);
						glEnd();
					}
				}
				else
				{
					glColor3d(Muscle[muscleNumber].color.x, Muscle[muscleNumber].color.y, Muscle[muscleNumber].color.z);
					glBegin(GL_LINES);
						glVertex3f(Node[i].position.x, Node[i].position.y, Node[i].position.z);
						glVertex3f(Node[k].position.x, Node[k].position.y, Node[k].position.z);
					glEnd();
				}
			}
		}	
	}
	
	if(MouseFunctionOnOff == 1)
	{
		//glColor3d(1.0, 1.0, 1.0);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    		glEnable(GL_DEPTH_TEST);
		glColor4f(1.0, 1.0, 1.0, 1.0);
		glPushMatrix();
		glTranslatef(MouseX, MouseY, MouseZ);
		
		glutSolidSphere(HitMultiplier*RadiusOfAtria,20,20);
		//glutSolidSphere(5.0*NodeRadiusAdjustment*RadiusOfAtria,20,20);
		glPopMatrix();
		glDisable(GL_BLEND);
	}

	glutSwapBuffers();
	
	if(MovieOn == 1)
	{
		glReadPixels(5, 5, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, Buffer);
		fwrite(Buffer, sizeof(int)*XWindowSize*YWindowSize, 1, MovieFile);
	}
}
