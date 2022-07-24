def test_post_store(client):
    """
    Test that the store endpoint returns the correct data.
    """

    request = {
        "DayOfWeek": 4,
        "Date": "2015-09-17",
        "Open": True,
        "Promo": True,
        "StateHoliday": False,
        "SchoolHoliday": False,
    }
    store_id = 4
    response = client.post(f"/sales/store/{store_id}", json=request)
    json_response = response.get_json()
    assert response.status_code == 200

    assert json_response["Store"] == store_id
    assert json_response["Date"] == request["Date"]
