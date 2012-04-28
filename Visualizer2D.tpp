//#include "Visualizer2D.h" 
//
//
//// function pointers for PBO Extension
//// Windows needs to get function pointers from ICD OpenGL drivers,
//// because opengl32.dll does not support extensions higher than v1.1.
//#if defined (_WIN32) || defined (_WIN64)
//PFNGLGENBUFFERSARBPROC pglGenBuffersARB = 0;                     // PBO Name Generation Procedure
//PFNGLBINDBUFFERARBPROC pglBindBufferARB = 0;                     // PBO Bind Procedure
//PFNGLBUFFERDATAARBPROC pglBufferDataARB = 0;                     // PBO Data Loading Procedure
//PFNGLBUFFERSUBDATAARBPROC pglBufferSubDataARB = 0;               // PBO Sub Data Loading Procedure
//PFNGLDELETEBUFFERSARBPROC pglDeleteBuffersARB = 0;               // PBO Deletion Procedure
//PFNGLGETBUFFERPARAMETERIVARBPROC pglGetBufferParameterivARB = 0; // return various parameters of PBO
//PFNGLMAPBUFFERARBPROC pglMapBufferARB = 0;                       // map PBO procedure
//PFNGLUNMAPBUFFERARBPROC pglUnmapBufferARB = 0;                   // unmap PBO procedure
//#define glGenBuffersARB           pglGenBuffersARB
//#define glBindBufferARB           pglBindBufferARB
//#define glBufferDataARB           pglBufferDataARB
//#define glBufferSubDataARB        pglBufferSubDataARB
//#define glDeleteBuffersARB        pglDeleteBuffersARB
//#define glGetBufferParameterivARB pglGetBufferParameterivARB
//#define glMapBufferARB            pglMapBufferARB
//#define glUnmapBufferARB          pglUnmapBufferARB
//#endif
//
//
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::Visualizer2D()
//{
//	w_width = 512;
//	w_height = 512;
//
//	state_display = 0;
//	boundary_display = true;
//
//	//makeImpulse = false;
//
//	PBO_DISP_CUDA_resource = NULL;
//}
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::Visualizer2D(int window_width, int window_height, int dispResX, int dispResY)
//{
//	w_width = window_width;
//	w_height = window_height;
//
//	dispResolutionX = dispResX;
//	dispResolutionY = dispResY;
//
//	state_display = 0;
//	boundary_display = true;
//
//	//makeImpulse = false;
//
//	PBO_DISP_CUDA_resource = NULL;
//	// The rest of the members will be initialized
//	// once the driver is set, and the opengl context launched
//}
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::~Visualizer2D()
//{
//	param->clean();
//	glDeleteTextures(1, &textureId);
//	//delete PBO_DISP_data_d;
//	cudaGraphicsUnregisterResource(PBO_DISP_CUDA_resource);
//	glDeleteBuffersARB(1, &dispPBO);
//}
//
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//inline void Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::setParam(pdeParam &pdeParameters)
//{
//	param = &pdeParameters;
//}
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//inline void Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::setSolvers(Riemann_h horizontalSolver, Riemann_v verticalSolver/*, Limiter phiLimiter/**/)
//{
//	horizontal_solver = horizontalSolver;
//	vertical_solver = verticalSolver;
//}
///*float Visualizer2D::findAbsMax(Float* data, int width, int height, int paddedWidth, int numBoundCellsX, int numBoundCellsY)
//{
//	// we have a choice to include or not the boundary cells in search for the absolute max, of the values
//	// The default will be kept as including the boundary cells.
//	float maximum = -FLT_MAX;
//
//	for( int j = 0; j < height-1; j++)
//	{
//		for( int i = 0; i < width; i++)
//		{
//			if (absoluteValue(data[i + j*paddedWidth]) > maximum)
//			{
//				maximum = absoluteValue(data[i + j*paddedWidth]);
//			}
//		}
//	}
//	return maximum;
//}
//float Visualizer2D::setRange()
//{
//	Float * data_cpu = (Float*)malloc(paddedWidth*height*sizeof(Float));
//
//	cudaMemcpy(data_cpu, driver->params.gpuQ, paddedWidth*height*sizeof(Float), cudaMemcpyDeviceToHost);	// This is inefficient but it is done once.
//
//	float halfRange = findAbsMax(data_cpu, width, height, paddedWidth, driver->params.numBoundaryCellsX, driver->params.numBoundaryCellsY);
//
//	free(data_cpu);
//
//	if (halfRange == 0)
//		return 0.5;
//	else
//		return halfRange;
//}*/
//
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//inline void Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::getOGLPos(int mouseX, int mouseY, GLdouble &coordX, GLdouble &coordY, GLdouble &coordZ)
//{
//	GLint viewport[4];
//	GLdouble modelview[16];
//	GLdouble projection[16];
//	GLfloat winX, winY, winZ;
//	GLdouble posX, posY, posZ;
//
//	glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
//	glGetDoublev( GL_PROJECTION_MATRIX, projection );
//	glGetIntegerv( GL_VIEWPORT, viewport );
//
//	winX = (GLfloat)mouseX;
//	winY = (GLfloat)viewport[3] - (GLfloat)mouseY;
//	glReadPixels( mouseX, int(winY), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ );
//
//	gluUnProject( winX, winY, winZ, modelview, projection, viewport, &coordX, &coordY, &coordZ);
//}
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//inline int Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::nextPowerOfTwo(int num1, int num2)
//{
//	int k = 0;						// Let's take an example
//	int s1 = num1; int s2 = num2;	//
//	while (num1 !=0)				// nextPowerOfTwo(4,13)
//	{								// the next power for 4 is 4, and the next power of 13 is 16, the answer must be 16
//		num1 = num1>>1;				// 
//		k++;						// for 4, k is 3,
//	}								// as [4->2 k->1], [2->1  k->2], [1->0 k->3] 
//	if ( (1<<(k-1)) - s1 == 0 )		// but 1<<2 is 4 and 4-4 is 0
//		k = k-1;					// then k(=3) becomes 2 (k-1=3-1=2)
//	int l = 0;						// 
//	while (num2 != 0)				// for 13, l is 4
//	{								// as [13->6 l->1], [6->3 k->2], [3->1 k->3], [1->0 k->4]
//		num2 = num2>>1;				// and 1<<4 = 16, 16-13 != 0
//		l++;						//
//	}								// maximum(2,4) = 4, the returned value is 1<<4 = 16. Done.
//	if ( (1<<(l-1)) - s2 == 0 )		//
//		l = l-1;
//	int maximum = (k>=l)? k:l;
//
//	return 1<<maximum;
//}
//// Check for PBO and NPOT texture support on the video card
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//inline void Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::checkGPUCompatibility()
//{
//	glInfo.getInfo();
//
//	supportPBO = false;
//	supportNPOT = false;
//#if defined (_WIN32) || defined (_WIN64)
//	// check PBO is supported by your video card
//	if(glInfo.isExtensionSupported("GL_ARB_pixel_buffer_object"))
//	{
//		// get pointers to GL functions
//		glGenBuffersARB		=		(PFNGLGENBUFFERSARBPROC)wglGetProcAddress("glGenBuffersARB");
//		glBindBufferARB		=		(PFNGLBINDBUFFERARBPROC)wglGetProcAddress("glBindBufferARB");
//		glBufferDataARB		=		(PFNGLBUFFERDATAARBPROC)wglGetProcAddress("glBufferDataARB");
//		glBufferSubDataARB	=		(PFNGLBUFFERSUBDATAARBPROC)wglGetProcAddress("glBufferSubDataARB");
//		glDeleteBuffersARB	=		(PFNGLDELETEBUFFERSARBPROC)wglGetProcAddress("glDeleteBuffersARB");
//		glGetBufferParameterivARB = (PFNGLGETBUFFERPARAMETERIVARBPROC)wglGetProcAddress("glGetBufferParameterivARB");
//		glMapBufferARB		=		(PFNGLMAPBUFFERARBPROC)wglGetProcAddress("glMapBufferARB");
//		glUnmapBufferARB	=		(PFNGLUNMAPBUFFERARBPROC)wglGetProcAddress("glUnmapBufferARB");
//
//		// check once again PBO extension
//		if(glGenBuffersARB && glBindBufferARB && glBufferDataARB && glBufferSubDataARB &&
//			glMapBufferARB && glUnmapBufferARB && glDeleteBuffersARB && glGetBufferParameterivARB)
//		{
//			supportPBO = true;
//			cout << "Video card supports GL_ARB_pixel_buffer_object." << endl;
//		}
//		else
//		{
//			supportPBO = false;
//			cout << "Video card does NOT support GL_ARB_pixel_buffer_object." << endl;
//		}
//	}
//	if (glInfo.isExtensionSupported("GL_ARB_texture_non_power_of_two"))
//	{
//		supportNPOT = true;
//		cout << "Video card supports GL_ARB_texture_non_power_of_two." << endl;
//	}
//#else // for linux, do not need to get function pointers, it is up-to-date
//	if(glInfo->isExtensionSupported("GL_ARB_pixel_buffer_object"))
//	{
//		supportPBO = true;
//		cout << "Video card supports GL_ARB_pixel_buffer_object." << endl;
//	}
//	else
//	{
//		supportPBO = false;
//		cout << "Video card does NOT support GL_ARB_pixel_buffer_object." << endl;
//	}
//	if (glInfo->isExtensionSupported("GL_ARB_texture_non_power_of_two"))
//	{
//		supportNPOT = true;
//		cout << "Video card supports GL_ARB_texture_non_power_of_two." << endl;
//	}
//#endif
//}
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//inline void Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::InitGlut(int &argc, char** argv)
//{
//	// Set display device
//	// Done at the beginning of main
//	//cudaError_t errorDevice = cudaSetDevice(0);
//	//cudaError_t errorGLdevice = cudaGLSetGLDevice(0);
//
//	// Initialise Glut
//	glutInit(&argc, argv);
//	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
//
//	// Create and position window
//	glutInitWindowPosition(100,100);
//	glutInitWindowSize(w_width,w_height);
//	glutCreateWindow("2D Visuals of PDE Solution");
//
//	// register callbacks
//	//
//	// Resize function
//	glutReshapeFunc(RESHAPE);
//
//	// Set functions for drawing
//	glutDisplayFunc(RENDER);
//
//	// Update through Idle function
//	glutIdleFunc(RENDER);
//
//	// Normal key press handler
//	glutKeyboardFunc(KEYPRESS);
//
//	// Mouse click handler
//	glutMouseFunc(MOUSECLICK);
//
//	glEnable(GL_DEPTH_TEST);    //Makes 3D drawing work when something is in front of something else
//	glEnable(GL_NORMALIZE);		//Normalizes vectors that should be normalized
//	glEnable(GL_TEXTURE_2D);	//Enables 2D textures
//	glEnable(GL_CULL_FACE);		//Culls the back face by default
//
//	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
//	glEnable(GL_COLOR_MATERIAL);
//
//	glClearColor(0.2f,0.15f,0.2f,1.0f);	// Background color
//}
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//inline void Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::initializeDisplay()
//{
//	// Set Canvas ratio
//	canvas_ratio = (float)(param->cellsX) / (float)(param->cellsY);
//	centerX = 0.0f;
//	centerY = 0.0f;
//	centerZ = 0.0f;
//
//	// Texture preparations
//	glGenTextures(1, &textureId);
//	glBindTexture(GL_TEXTURE_2D, textureId);
//
//	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // can be set to GL_NEAREST
//	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // can be set to GL_NEAREST
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // can be set to GL_LINEAR
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); // can be set to GL_LINEAR
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
//	glBindTexture(GL_TEXTURE_2D, 0);
//
//	checkGPUCompatibility();
//
//	// Do different things for different supports from the GPU
//	// Find next largest power of two of both dimensions
//	if (!supportNPOT)
//	{
//		int nextPowerOfTwo = Visualizer2D::nextPowerOfTwo(dispResolutionX, dispResolutionY);
//
//		dispResolutionX = nextPowerOfTwo;
//		dispResolutionY = nextPowerOfTwo;
//	}
//
//	displaySize = dispResolutionX * dispResolutionY * sizeof(float);
//
//	//range = setRange();
//
//	// WARNING! By this point we should have started an OpenGL context
//	// Display PBO
//	glGenBuffersARB(1, &dispPBO);
//	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, dispPBO);
//	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, displaySize, 0, GL_STREAM_DRAW_ARB);	// memory allocation for the buffer
//
//	cudaError_t ret = cudaGraphicsGLRegisterBuffer(&PBO_DISP_CUDA_resource, dispPBO, cudaGraphicsMapFlagsWriteDiscard);
//	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
//}
//
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//inline void Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::reshapeWindow(int w, int h)
//{
//	// Prevent a divide by zero, when window is too short
//	// (you cant make a window of zero height).
//	if (h == 0)
//		h = 1;
//	float ratio =  w * 1.0 / h;
//
//	// Use the Projection Matrix
//	glMatrixMode(GL_PROJECTION);
//
//	// Reset Matrix
//	glLoadIdentity();
//
//	// Set the viewport to be the entire window
//	glViewport(0, 0, w, h);
//
//	// Set the correct perspective.
//	gluPerspective(45.0f, ratio, 0.1f, 100.0f);
//
//	// Get Back to the Modelview
//	glMatrixMode(GL_MODELVIEW);
//	gluLookAt(0.0,  0.0,   2.0,		//The camera position
//			  0.0,  0.0,  -5.0,		//The point we're looking at
//			  0.0f, 1.0f,  0.0f);	//The up vector
//}
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//inline void Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::visualizePDE()
//{
//	checkChanges();	// check for any interactivity flag and apply changes, reset necessary flags to default.
//
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//	glMatrixMode(GL_MODELVIEW);
//	glLoadIdentity();
//	gluLookAt(0.0,  0.0,   2.0,		//The camera position
//			  0.0,  0.0,  -5.0,		//The point we're looking at
//			  0.0f, 1.0f,  0.0f);	//The up vector
//
//	/***********************************************Display and Update***********************************************/
//
//	cudaGraphicsMapResources(1, &PBO_DISP_CUDA_resource, 0);
//	cudaGraphicsResourceGetMappedPointer((void **)&PBO_DISP_data_d, &displaySize, PBO_DISP_CUDA_resource);
//
//	// This function can be modified to put the read data into a format that can be used for colored mapping
//	copyDisplayData(PBO_DISP_data_d, dispResolutionX, dispResolutionY, param->qNew, param->cellsX, param->cellsY, param->numStates, param->ghostCells, state_display, boundary_display);
//
//	// would be useable if double texture is supported, and boundaries are to be included:		// turns out not really, when padding is on
//	// Not just padding, different sizes of resolution and data size...
//	//cudaMemcpy(PBO_DISP_data_d, param->qNew, displaySize, cudaMemcpyDeviceToDevice);
//
//	cudaGraphicsUnmapResources(1, &PBO_DISP_CUDA_resource, 0);
//
//	glBindBufferARB( GL_PIXEL_UNPACK_BUFFER_ARB, dispPBO);
//	glBindTexture(GL_TEXTURE_2D, textureId); //Tell OpenGL which texture to edit, I only need to do this once, and it is done in the intialisations
//
//	glTexImage2D(	GL_TEXTURE_2D,
//					0,
//					GL_LUMINANCE,
//					dispResolutionX,	//(disp_width+2*driver->params.numBoundaryCellsX),
//					dispResolutionY,	//(disp_height+2*driver->params.numBoundaryCellsY),
//					0,
//					GL_LUMINANCE,
//					GL_FLOAT,
//					0);
//
//	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
//
//	// do frame update
//	// find out the current frame we're on.
//	static int frame = 0;
//	frame++;
//
//	step<Riemann_h, Riemann_v>(*param, horizontal_solver, vertical_solver);
//
//	//***************************************************************************************************************//
//
//	drawCanvas();
//
//	glutSwapBuffers();
//}
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//inline void Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::drawCanvas()
//{
//	glTranslatef(0.0, 0.0, 0.0);
//	//canvas
//	glColor3f(1.0f, 1.0f, 1.0f);
//	glBegin(GL_QUADS);
//
//	glNormal3f(0.0f, 0.0f, 1.0f);
//
//	// very simple dimension adjustment, will fix the width dimension to 1 and adjust the height,
//	// might end up outside the screen, will require a better check, maybe switch between fixing
//	// width or height, and translate things to keep in the viewable zone.
//
//	glTexCoord2f(0.0f, 0.0f);												//
//	glVertex3f(centerX-0.5f, centerY-(1.0f/canvas_ratio)/2.0f, centerZ);	// define
//	// bottom
//	glTexCoord2f(1.0f, 0.0f);												// line		//
//	glVertex3f(centerX+0.5f, centerY-(1.0f/canvas_ratio)/2.0f, centerZ);	//			// define
//	// right
//	glTexCoord2f(1.0f, 1.0f);												//			// line
//	glVertex3f(centerX+0.5f, centerY+(1.0f/canvas_ratio)/2.0f, centerZ);	// define	//
//	// upper
//	glTexCoord2f(0.0f, 1.0f);												// line
//	glVertex3f(centerX-0.5f, centerY+(1.0f/canvas_ratio)/2.0f, centerZ);	//
//
//	glEnd();
//}
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//inline void Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::checkChanges()
//{/*
//	if (makeImpulse)
//	{	
//		
//		doImpulse(impulseX, impulseY, impulseZ, centerX, centerY, centerZ, canvas_ratio,
//			driver->params.gpuQ,
//			driver->params.calculateXCells(),	// returns padded number
//			width,								// returns unpadded number
//			height,
//			driver->params.numBoundaryCellsX,
//			driver->params.numBoundaryCellsY,
//			state_display);					// given the impulse coords the deimensions of the canvas, the given state
//		*/
//		/*
//		// suggestion, make inside visualizer2D.h under canvas,
//		a center for the canvas and the width and hight of the canvas,
//		later in init display we can assign values to these and use them in draw canvas,
//		we can then use these values and the impulse coordinates to determine
//		the exact location where the impulse should be, mainly by getting the ration of where the hit was,
//		and finding the most appropriate location corresponding to it
//		in the computational data and making a kind of impulse there.
//		*/
//	/*
//		makeImpulse = false;
//	}*/
//}
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//inline void Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::normalKeyPress(unsigned char key, int x, int y)
//{
//	if (key == 27)	// escape
//	{
//		delete visualizer;
//		cudaThreadExit();
//		exit(0);
//	}
//	int maxNumStates = param->numStates;
//	switch(key)
//	{
//	case '1':
//		if (maxNumStates > 0 ) state_display = 0; break;
//	case '2':
//		if (maxNumStates > 1 ) state_display = 1; break;
//	case '3':
//		if (maxNumStates > 2 ) state_display = 2; break;
//	case '4':
//		if (maxNumStates > 3 ) state_display = 3; break;
//	case '5':
//		if (maxNumStates > 4 ) state_display = 4; break;
//	case '6':
//		if (maxNumStates > 5 ) state_display = 5; break;
//	case '7':
//		if (maxNumStates > 6 ) state_display = 6; break;
//	case '8':
//		if (maxNumStates > 7 ) state_display = 7; break;
//	case '9':
//		if (maxNumStates > 8 ) state_display = 8; break;
//	case '0':
//		if (maxNumStates > 9 ) state_display = 9; break;
//
//	case 'b':
//	case 'B':
//		boundary_display = !boundary_display;break;
//
//	default:
//		break;
//	}
//}
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//inline void Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::mouseClick(int button, int state, int x, int y)
//{
//	if (state == GLUT_DOWN)
//	{
//		getOGLPos(x, y, impulseX, impulseY, impulseZ);
//
//		if (button == GLUT_LEFT_BUTTON)
//		{
//			//makeImpulse = true;
//		}
//	}
//	else // if (state == GLUT_UP)
//	{
//	}
//}
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//inline void Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::RESHAPE(int w, int h)
//{
//	visualizer->reshapeWindow(w, h);
//}
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//inline void Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::RENDER()
//{
//	visualizer->visualizePDE();
//}
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//inline void Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::KEYPRESS(unsigned char key, int x, int y)
//{
//	visualizer->normalKeyPress(key, x, y);
//}
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//inline void Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::MOUSECLICK(int button, int state, int x, int y)
//{
//	visualizer->mouseClick(button, state, x, y);
//}
//template<class Riemann_h, class Riemann_v/*, class Limiter/**/>
//inline void Visualizer2D<Riemann_h, Riemann_v/*, class Limiter/**/>::launchDisplay()
//{
//	glutMainLoop();
//}