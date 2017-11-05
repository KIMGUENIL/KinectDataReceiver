// RecordHelpDlg.cpp : ���� �����Դϴ�.
//

#include "stdafx.h"
#include "KinectVideoCapture.h"
#include "RecordHelpDlg.h"
#include "afxdialogex.h"
#include <string.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include "cv.h"
using namespace cv;
using namespace std;


#ifdef _DEBUG 
#pragma comment (lib, "opencv_world320d.lib") 
#else 
#pragma comment (lib, "opencv_world320.lib")
#endif
// CRecordDlg ��ȭ �����Դϴ�.

// RecordHelpDlg ��ȭ �����Դϴ�.

IMPLEMENT_DYNAMIC(RecordHelpDlg, CDialogEx)

RecordHelpDlg::RecordHelpDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_RECORD, pParent)
	, m_sex(FALSE)
{

}

RecordHelpDlg::~RecordHelpDlg()
{
}

void RecordHelpDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_PLACE_LIST, m_place);
	DDX_Control(pDX, IDC_PLACE_LIST2, m_stuff);
	DDX_Radio(pDX, IDC_RADIO1, m_sex);
}


BEGIN_MESSAGE_MAP(RecordHelpDlg, CDialogEx)
	ON_WM_TIMER()
	ON_WM_DESTROY()
	ON_BN_CLICKED(IDC_OK_BUTTON, &RecordHelpDlg::OnBnClickedOkButton)
END_MESSAGE_MAP()


// RecordHelpDlg �޽��� ó�����Դϴ�.


BOOL RecordHelpDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	//��� �ҷ�����
	ifstream inputFile("place.txt");

	if (inputFile.is_open()) {
		while (!inputFile.eof()) {
			string p;
			inputFile >> p;

			CString cstr(p.c_str());
			m_place.AddString(cstr);
		}
	}

	inputFile.close();
	//���� �ҷ�����
	inputFile.open("stuff.txt");

	if (inputFile.is_open()) {
		while (!inputFile.eof()) {
			string p;
			inputFile >> p;

			CString cstr(p.c_str());
			m_stuff.AddString(cstr);
		}
	}

	inputFile.close();

	return TRUE;  // return TRUE unless you set the focus to a control
				  // ����: OCX �Ӽ� �������� FALSE�� ��ȯ�ؾ� �մϴ�.
}


void RecordHelpDlg::OnDestroy()
{
	CDialogEx::OnDestroy();

	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰��մϴ�.
}


void RecordHelpDlg::saveBehavior() {
	CString placeCStr;
	CString stuffCStr;

	CListBox *p_list = (CListBox *)GetDlgItem(IDC_PLACE_LIST);

	int index = p_list->GetCurSel();
	if (index != LB_ERR) {
		p_list->GetText(index, placeCStr);
	}

	CListBox *s_list = (CListBox *)GetDlgItem(IDC_PLACE_LIST2);

	index = s_list->GetCurSel();
	if (index != LB_ERR) {
		s_list->GetText(index, stuffCStr);
	}

	for (int i = 0; i < 55; i++) {
		BOOL chk = IsDlgButtonChecked(1200 + i);

		if (chk) {
			if (behavior == "") {
				behavior = to_string(i+1);
			}
			else {
				behavior = behavior + ", " + to_string(i+1);
			}
		}
	}

	CT2CA pCSs(placeCStr);
	placeStr = pCSs;

	CT2CA sCSs(stuffCStr);
	stuffStr = sCSs;

	UpdateData();
	if (m_sex == 0) {
		sex = "����";
	}
	else {
		sex = "����";
	}

	CString CAge;
	GetDlgItem(IDC_EDIT1)->GetWindowText(CAge);
	CT2CA aCSs(CAge);
	age = aCSs;
}

void RecordHelpDlg::OnBnClickedOkButton()
{
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�.
	saveBehavior();

	::SendMessage(this->m_hWnd, WM_CLOSE, NULL, NULL);

	RecordHelpDlg::OnOK();
}
