#pragma once
#include "afxwin.h"
#include "opencv.hpp"
#include "app.h"
#include "highgui.hpp"
#include "cv.h"
#include "KinectVideoCaptureDlg.h"
#include <string.h>
#include <fstream>
// CRecordDlg ��ȭ �����Դϴ�.


// RecordHelpDlg ��ȭ �����Դϴ�.

class RecordHelpDlg : public CDialogEx
{
	DECLARE_DYNAMIC(RecordHelpDlg)

public:
	RecordHelpDlg(CWnd* pParent = NULL);   // ǥ�� �������Դϴ�.
	virtual ~RecordHelpDlg();

// ��ȭ ���� �������Դϴ�.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_RECORD };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV �����Դϴ�.

	DECLARE_MESSAGE_MAP()

public:
	CListBox m_place;
	CListBox m_stuff;
	virtual BOOL OnInitDialog();
	afx_msg void OnDestroy();

	void saveBehavior();
	afx_msg void OnBnClickedOkButton();
	int m_sex;

	std::string stuffStr = "";
	std::string placeStr = "";
	std::string age = "";
	std::string sex = "";
	std::string behavior = "";
};
