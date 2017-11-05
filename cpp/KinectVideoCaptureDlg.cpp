// KinectVideoCaptureDlg.cpp : ���� ����

#include "stdafx.h"
#include "KinectVideoCapture.h"
#include "KinectVideoCaptureDlg.h"
#include "afxdialogex.h"
#include "app.h"
#include "RecordHelpDlg.h"
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include "util.h"
#include <direct.h>

#define RECODE_START 0
#define RECODE_END 1
#define RECODE_NOTHING 2

using namespace std;
#ifdef _DEBUG
#define new DEBUG_NEW
#endif

UINT RecodeThreadFunc(LPVOID pParam);

// ���� ���α׷� ������ ���Ǵ� CAboutDlg ��ȭ �����Դϴ�.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

	// ��ȭ ���� �������Դϴ�.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV �����Դϴ�.

// �����Դϴ�.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CKinectVideoCaptureDlg ��ȭ ����
CKinectVideoCaptureDlg::CKinectVideoCaptureDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_KINECTVIDEOCAPTURE_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CKinectVideoCaptureDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_PICTURE, m_picture);
	DDX_Control(pDX, IDC_PICTURE2, m_picture2);
}

BEGIN_MESSAGE_MAP(CKinectVideoCaptureDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_DESTROY()
	//	ON_WM_TIMER()
	ON_BN_CLICKED(IDC_BUTTON1, &CKinectVideoCaptureDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON3, &CKinectVideoCaptureDlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON2, &CKinectVideoCaptureDlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_PORT_BUTTON, &CKinectVideoCaptureDlg::OnBnClickedPortButton)
	ON_WM_TIMER()
END_MESSAGE_MAP()


// CKinectVideoCaptureDlg �޽��� ó����

BOOL CKinectVideoCaptureDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// �ý��� �޴��� "����..." �޴� �׸��� �߰��մϴ�.

	// IDM_ABOUTBOX�� �ý��� ��� ������ �־�� �մϴ�.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// �� ��ȭ ������ �������� �����մϴ�.  ���� ���α׷��� �� â�� ��ȭ ���ڰ� �ƴ� ��쿡��
	//  �����ӿ�ũ�� �� �۾��� �ڵ����� �����մϴ�.
	SetIcon(m_hIcon, TRUE);			// ū �������� �����մϴ�.
	SetIcon(m_hIcon, FALSE);		// ���� �������� �����մϴ�.

	// TODO: ���⿡ �߰� �ʱ�ȭ �۾��� �߰��մϴ�
	cKinect.initialize(0);
	dKinect.initialize(1);
	bKinect.initialize(2);

	m_picture.GetClientRect(&r);
	winSize.width = r.right;
	winSize.height = r.bottom;

	m_picture2.GetClientRect(&r2);
	winSize2.width = r2.right;
	winSize2.height = r2.bottom;

	pMog2 = createBackgroundSubtractorMOG2();

	wCount = 0;
	isAutoRecord = false;
	isRecording = false;

	GetDlgItem(IDC_BUTTON1)->EnableWindow(TRUE);
	GetDlgItem(IDC_BUTTON3)->EnableWindow(FALSE);

	TCHAR szBuf[MAX_PATH] = { 0, };
	CString strINIPath(_T("./config.ini"));
	CString strSection(_T("Video"));
	CString strDrawRate(_T("DRAWRATE"));
	CString strPartRate(_T("PARTRATE"));
	CString strStartTime(_T("STARTTIME"));
	CString strEndTime(_T("ENDTIME"));
	CString strSaveRoute(_T("SAVEROUTE"));

	GetPrivateProfileString(strSection, strDrawRate, _T(""), szBuf, MAX_PATH, strINIPath);
	valDrawRate.Format(_T("%s"), szBuf);
	GetPrivateProfileString(strSection, strPartRate, _T(""), szBuf, MAX_PATH, strINIPath);
	valPartRate.Format(_T("%s"), szBuf);
	GetPrivateProfileString(strSection, strStartTime, _T(""), szBuf, MAX_PATH, strINIPath);
	valStartTime.Format(_T("%s"), szBuf);
	GetPrivateProfileString(strSection, strEndTime, _T(""), szBuf, MAX_PATH, strINIPath);
	valEndTime.Format(_T("%s"), szBuf);
	GetPrivateProfileString(strSection, strSaveRoute, _T(""), szBuf, MAX_PATH, strINIPath);
	valSaveRoute.Format(_T("%s"), szBuf);

	CT2CA chSaveRoute(valSaveRoute);
	saveRoute = chSaveRoute;

	startTime = _ttoi(valStartTime);
	endTime = _ttoi(valEndTime);
	drawRate = 1000 / (_ttoi(valDrawRate));
	partRate = (_ttoi(valPartRate)) * 1000 * 60;

	cout << "���� �ð� : " << startTime << "��" << endl;
	cout << "���� �ð� : " << endTime << "��" << endl;
	cout << "ȭ�� ��� Frame : " << _ttoi(valDrawRate) << "������" << endl;
	cout << "���� ���� ���� ���� : " << _ttoi(valPartRate) << "��" << endl;
	cout << "���� ���� ��ġ : " << saveRoute << endl;

	_mkdir(saveRoute.c_str());
	_mkdir((saveRoute + "/Place").c_str());
	_mkdir((saveRoute + "/Stuff").c_str());
	_mkdir((saveRoute + "/Place/Color").c_str());
	_mkdir((saveRoute + "/Place/Depth").c_str());
	_mkdir((saveRoute + "/Place/Body").c_str());
	_mkdir((saveRoute + "/Stuff/Color").c_str());
	_mkdir((saveRoute + "/Stuff/Depth").c_str());
	_mkdir((saveRoute + "/Place/Body").c_str());

	drawTime = 0;
	partTime = 0;
	part = 0;
	fps = 20;
	nDelay = 1000 / fps;
	m_iRecod = RECODE_NOTHING;

	cimage_mfc = NULL;
	cimage2_mfc = NULL;

	sleepTime = false;

	SetTimer(0, nDelay, NULL);
	return TRUE;  // ��Ŀ���� ��Ʈ�ѿ� �������� ������ TRUE�� ��ȯ�մϴ�.
}

void CKinectVideoCaptureDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// ��ȭ ���ڿ� �ּ�ȭ ���߸� �߰��� ��� �������� �׸�����
//  �Ʒ� �ڵ尡 �ʿ��մϴ�.  ����/�� ���� ����ϴ� MFC ���� ���α׷��� ��쿡��
//  �����ӿ�ũ���� �� �۾��� �ڵ����� �����մϴ�.

void CKinectVideoCaptureDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // �׸��⸦ ���� ����̽� ���ؽ�Ʈ�Դϴ�.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Ŭ���̾�Ʈ �簢������ �������� ����� ����ϴ�.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// �������� �׸��ϴ�.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// ����ڰ� �ּ�ȭ�� â�� ���� ���ȿ� Ŀ���� ǥ�õǵ��� �ý��ۿ���
//  �� �Լ��� ȣ���մϴ�.
HCURSOR CKinectVideoCaptureDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CKinectVideoCaptureDlg::OnDestroy()
{
	CDialogEx::OnDestroy();

	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰��մϴ�.
	cKinect.finalize();
	dKinect.finalize();
}

void CKinectVideoCaptureDlg::OnTimer(UINT_PTR nIDEvent)
{
	tic();
	getTime();
	CTime cTime = CTime::GetCurrentTime();

	if (cTime.GetHour() == endTime) {
		if (isRecording) {
			RecordEnd();
			isRecording = false;
			part = 0;

			SetDlgItemText(IDC_RECORD_STATIC, _T("��ȭ ����"));
			GetDlgItem(IDC_BUTTON1)->EnableWindow(TRUE);
			GetDlgItem(IDC_BUTTON3)->EnableWindow(FALSE);
		}

		KillTimer(0);
		SetTimer(0, nDelay, NULL);
	}
	else {
		if (cTime.GetHour() == startTime) {
			if (!isRecording) {
				isRecording = true;

				CT2CA videoTime(mDay + "_" + mTime);
				string videoName = videoTime;
				RecordStart(videoName, to_string(part));

				SetDlgItemText(IDC_RECORD_STATIC, _T("��ȭ ��"));
				GetDlgItem(IDC_BUTTON1)->EnableWindow(FALSE);
				GetDlgItem(IDC_BUTTON3)->EnableWindow(TRUE);
			}
		}
		cKinect.updateColor();
		dKinect.updateDepth();
		dKinect.updateBody();
		dKinect.drawDepth();
		cKinect.drawColor();
		dKinect.drawBody();

		if (drawTime >= drawRate) {
			resize(cKinect.colorMat, cmat_frame, Size(480, 320));

			//BackGroundSubtraction
			//pMog2->apply(cmat_frame, bg);

			cimage_mfc = new CImage();
			cimage2_mfc = new CImage();
			cimage_mfc->Create(winSize.width, winSize.height, 24);
			cimage2_mfc->Create(winSize2.width, winSize2.height, 32);

			BITMAPINFO bitInfo;
			bitInfo.bmiHeader.biBitCount = 32;
			bitInfo.bmiHeader.biWidth = cmat_frame.cols;
			bitInfo.bmiHeader.biHeight = -cmat_frame.rows;
			bitInfo.bmiHeader.biPlanes = 1;
			bitInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
			bitInfo.bmiHeader.biCompression = BI_RGB;
			bitInfo.bmiHeader.biClrImportant = 0;
			bitInfo.bmiHeader.biClrUsed = 0;
			bitInfo.bmiHeader.biSizeImage = 0;
			bitInfo.bmiHeader.biXPelsPerMeter = 0;
			bitInfo.bmiHeader.biYPelsPerMeter = 0;

			int destx = 0, desty = 0;
			int destw = winSize.width;
			int desth = winSize.height;
			int imgx = 0, imgy = 0;
			int imgWidth = cmat_frame.cols;
			int imgHeight = cmat_frame.rows;

			StretchDIBits(cimage_mfc->GetDC(),
				destx, desty, destw, desth,
				imgx, imgy, imgWidth, imgHeight,
				cmat_frame.data, &bitInfo, DIB_RGB_COLORS, SRCCOPY);

			HDC dc = ::GetDC(m_picture.m_hWnd);
			cimage_mfc->BitBlt(dc, 0, 0);
			::ReleaseDC(m_picture.m_hWnd, dc);
			cimage_mfc->ReleaseDC();
			cimage_mfc->Destroy();

			//////////////////////////////////// Depth ////////////////////////////////////////////////

			resize(dKinect.depthMat, dmat_frame, Size(480, 320));
			dmat_frame.convertTo(dmat_frame, CV_8U, -255.0 / 8000.0, 255.0);
			cvtColor(dmat_frame, dmat_frame, COLOR_GRAY2BGR);

			BITMAPINFO bitinfo2;
			bitinfo2.bmiHeader.biBitCount = 24;
			bitinfo2.bmiHeader.biWidth = dmat_frame.cols;
			bitinfo2.bmiHeader.biHeight = -dmat_frame.rows;
			bitinfo2.bmiHeader.biPlanes = 1;
			bitinfo2.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
			bitinfo2.bmiHeader.biCompression = BI_RGB;
			bitinfo2.bmiHeader.biClrImportant = 0;
			bitinfo2.bmiHeader.biClrUsed = 0;
			bitinfo2.bmiHeader.biSizeImage = 0;
			bitinfo2.bmiHeader.biXPelsPerMeter = 0;
			bitinfo2.bmiHeader.biYPelsPerMeter = 0;

			StretchDIBits(cimage2_mfc->GetDC(),
				destx, desty, destw, desth,
				imgx, imgy, imgWidth, imgHeight,
				dmat_frame.data, &bitinfo2, DIB_RGB_COLORS, SRCCOPY);

			HDC dc2 = ::GetDC(m_picture2.m_hWnd);
			cimage2_mfc->BitBlt(dc2, 0, 0);
			::ReleaseDC(m_picture2.m_hWnd, dc2);
			cimage2_mfc->ReleaseDC();
			cimage2_mfc->Destroy();

			drawTime = 0;
		}

		if (m_iRecod == RECODE_START) {
			inQFrame();
		}

		if (isRecording) {
			partTime += nDelay;

			if (partTime >= partRate) {
				part++;
				cout << "���� " << part << "��°" << endl;

				RecordEnd();

				CT2CA videoTime(mDay + "_" + mTime);
				string videoName = videoTime;
				RecordStart(videoName, to_string(part));
				partTime = 0;
			}
		}

		double delay = nDelay - toc();
		drawTime += nDelay;

		KillTimer(0);
		SetTimer(0, (delay > 0) ? delay : 1, NULL);
	}

	CDialogEx::OnTimer(nIDEvent);
}

void CKinectVideoCaptureDlg::OnBnClickedButton1()
{
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�
	stuffStr = "";
	placeStr = "";
	age = "";
	sex = "";
	behavior = "";

	RecordHelpDlg crDlg;

	if (IDOK == crDlg.DoModal()) {
		stuffStr = crDlg.stuffStr;
		placeStr = crDlg.placeStr;
		age = crDlg.age;
		sex = crDlg.sex;
		behavior = crDlg.behavior;
	}

	isRecording = true;
	CT2CA videoTime(mDay + "_" + mTime);
	string videoName = videoTime;
	RecordStart(videoName, to_string(part));

	SetDlgItemText(IDC_RECORD_STATIC, _T("��ȭ ��"));

	GetDlgItem(IDC_BUTTON1)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON3)->EnableWindow(TRUE);
}

void CKinectVideoCaptureDlg::OnBnClickedButton3()
{
	RecordEnd();

	isRecording = false;
	part = 0;

	SetDlgItemText(IDC_RECORD_STATIC, _T("��ȭ ����"));

	GetDlgItem(IDC_BUTTON1)->EnableWindow(TRUE);
	GetDlgItem(IDC_BUTTON3)->EnableWindow(FALSE);
}

void CKinectVideoCaptureDlg::getTime()
{
	CTime cTime = CTime::GetCurrentTime();
	mDay.Format(L"%d.%d.%d", cTime.GetYear(), cTime.GetMonth(), cTime.GetDay());
	mTime.Format(L"%d�� %d�� %d��", cTime.GetHour(), cTime.GetMinute(), cTime.GetSecond());

	SetDlgItemText(IDC_TimeSTATIC, mDay + "\n" + mTime);
}

int CKinectVideoCaptureDlg::getRate()
{
	int count = 0;

	for (int i = 0; i < bg.rows; i++) {
		for (int j = 0; j < bg.cols; j++) {
			if (bg.at<uchar>(i, j) == 255) {
				count++;
			}
		}
	}

	return count;
}

void CKinectVideoCaptureDlg::OnBnClickedButton2()
{
	// ������ read
	CString CRate;
	GetDlgItem(IDC_EDIT1)->GetWindowText(CRate);
	double rate = _wtof(CRate);

	//��ȭ control
	wCount = rate * bg.rows * bg.cols;

	GetDlgItem(IDC_BUTTON1)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON3)->EnableWindow(FALSE);

	CString btnName;

	GetDlgItem(IDC_BUTTON2)->GetWindowText(btnName);

	if (btnName == "����") {
		GetDlgItem(IDC_BUTTON2)->SetWindowText(_T("��ȭ ��"));
		isAutoRecord = true;
	}
	else {
		GetDlgItem(IDC_BUTTON2)->SetWindowText(_T("����"));
		isAutoRecord = false;

		GetDlgItem(IDC_BUTTON1)->EnableWindow(TRUE);
		GetDlgItem(IDC_BUTTON3)->EnableWindow(FALSE);
	}
}

void CKinectVideoCaptureDlg::OnBnClickedPortButton()
{
	CString CRate;
	GetDlgItem(IDC_EDIT1)->GetWindowText(CRate);
	double rate = _wtof(CRate);

	//��ȭ control
	wCount = rate * bg.rows * bg.cols;

	GetDlgItem(IDC_BUTTON1)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON3)->EnableWindow(FALSE);

	CString temp;

	GetDlgItem(IDC_STR_PORT)->GetWindowText(temp);
	m_strPort = _wtoi(temp);

	servSock.Create(m_strPort);
	servSock.Listen();

	if (servSock.Accept(clintSock)) {
		MessageBox(L"���� ����");
		isAutoRecord = true;
	}
}

void CKinectVideoCaptureDlg::tic() {
	QueryPerformanceFrequency(&liFrequency);
	QueryPerformanceCounter(&liCounter1);
}

double CKinectVideoCaptureDlg::toc() {
	double ret = 0;
	QueryPerformanceCounter(&liCounter2);
	ret = (double)(liCounter2.QuadPart - liCounter1.QuadPart) / (double)liFrequency.QuadPart * 1000;

	return ret;
}

void CKinectVideoCaptureDlg::inQFrame() {
	RECODE_FORMAT rec;
	memset(&rec, 0, sizeof(RECODE_FORMAT));

	IplImage cFrame = cKinect.colorMat;
	IplImage dFrame = dKinect.depthMat;

	rec.width = cFrame.width;
	rec.imageSize = cFrame.imageSize;
	rec.height = cFrame.height;
	rec.pImageData = (char *)calloc(cFrame.imageSize, 1);

	rec.dWidth = dFrame.width;
	rec.dHeight = dFrame.height;
	rec.dImageSize = dFrame.imageSize;
	rec.dpImageData = (char *)calloc(dFrame.imageSize, 1);

	memcpy(rec.pImageData, cFrame.imageData, cFrame.imageSize);
	memcpy(rec.dpImageData, dFrame.imageData, dFrame.imageSize);

	m_ctRecode.Lock();
	m_RecodFrameList.AddTail(rec);
	m_ctRecode.Unlock();
}

UINT RecodeThreadFunc(LPVOID pParam)
{
	CKinectVideoCaptureDlg* pMe = (CKinectVideoCaptureDlg *)pParam;
	BOOL		 bEmpty;
	while (1)
	{
		RECODE_FORMAT rec;
		POSITION pos = 0;

		memset(&rec, 0, sizeof(RECODE_FORMAT));

		pMe->m_ctRecode.Lock();
		bEmpty = pMe->m_RecodFrameList.IsEmpty();
		pMe->m_ctRecode.Unlock();

		if (bEmpty)
		{
			if (pMe->m_iRecod == RECODE_END)
			{
				pMe->cKinect.outputVideo.release();
				pMe->depthRaw.close();
				pMe->m_iRecod = RECODE_NOTHING;

				return 0;
			}

			Sleep(5);
			continue;
		}

		pMe->m_ctRecode.Lock();
		rec = pMe->m_RecodFrameList.GetHead();
		pMe->m_RecodFrameList.RemoveHead();
		pMe->m_ctRecode.Unlock();

		Mat pRecFrame = cv::Mat(rec.height, rec.width, CV_8UC4, rec.pImageData);
		cvtColor(pRecFrame, pRecFrame, CV_BGRA2BGR);

		pMe->cKinect.outputVideo.write(pRecFrame);
		pMe->depthRaw.write(rec.dpImageData, rec.dImageSize);
		pMe->depthRaw.flush();
		//cout << pMe->m_RecodFrameList.GetSize() << endl;

		free(rec.pImageData);
		free(rec.dpImageData);
	}


	return 0;
}

void CKinectVideoCaptureDlg::RecordStart(string stuffStr, string placeStr, string age, string sex, string behavior, string part) {
	m_ctRecode.Lock();
	if (stuffStr == "") {
		std::string videoName = sex + "_" + age + "_" + behavior + "_" + placeStr + "_" + part;

		cKinect.outputVideo.open(saveRoute + "Place/Color/" + videoName + ".avi", VideoWriter::fourcc('X', 'V', 'I', 'D'), 20, Size(1920, 1080), true);
		depthRaw.open(saveRoute + "Place/Depth/" + videoName + ".dat", std::ios_base::out | std::ios_base::binary);
	}

	else if (behavior == "") {
		std::string videoName = sex + "_" + age + "_" + stuffStr + "_" + placeStr + "_" + part;

		cKinect.outputVideo.open(saveRoute + "Stuff/Color/" + videoName + ".avi", VideoWriter::fourcc('X', 'V', 'I', 'D'), 20, Size(1920, 1080), true);
		depthRaw.open(saveRoute + "Place/Depth/" + videoName + ".dat", std::ios_base::out | std::ios_base::binary);
	}
	m_ctRecode.Unlock();

	AfxBeginThread(RecodeThreadFunc, this);
	m_iRecod = RECODE_START;
}

void CKinectVideoCaptureDlg::RecordStart(string time, string part) {
	m_ctRecode.Lock();

	std::string videoName = time + "_" + part;
	cKinect.outputVideo.open(saveRoute + "Place/Color/" + videoName + ".avi", VideoWriter::fourcc('X', 'V', 'I', 'D'), 20, Size(1920, 1080), true);
	depthRaw.open(saveRoute + "Place/Depth/" + videoName + ".dat", std::ios_base::out | std::ios_base::binary);

	m_ctRecode.Unlock();

	AfxBeginThread(RecodeThreadFunc, this);
	m_iRecod = RECODE_START;
}

void CKinectVideoCaptureDlg::RecordEnd() {
	m_iRecod = RECODE_END;

	while (m_iRecod != RECODE_NOTHING) {
		Sleep(5);
	}
}